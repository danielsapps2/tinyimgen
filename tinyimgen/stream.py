"""Stream a stack of layers through a small GPU from pinned host RAM.

Layer i+1 is copied on a side stream while layer i computes, so on PCIe 3.0
(~13 GB/s) the transfer is mostly hidden behind compute. Only ~2 layers are
resident at any time.
"""
import torch
from torch import nn


class LayerStreamer:
    def __init__(self, layers, device="cuda", cyclic=False):
        self.layers = list(layers)
        self.device = torch.device(device)
        self.cyclic = cyclic  # prefetch layer 0 after the last one (repeated denoise steps)
        self.copy_stream = torch.cuda.Stream(self.device)
        n = len(self.layers)
        self.entries, self.events, self.loaded = [], [None] * n, [False] * n
        self.done = [None] * n  # compute-finished events, used to throttle the host
        for i, layer in enumerate(self.layers):
            layer.to("cpu")
            ents = []
            for t in list(layer.parameters()) + list(layer.buffers()):
                t.data = t.data.pin_memory()
                ents.append((t, t.data))
            self.entries.append(ents)
            layer.register_forward_pre_hook(self._pre(i))
            layer.register_forward_hook(self._post(i))

    def _load(self, i):
        if self.loaded[i]:
            return
        with torch.cuda.stream(self.copy_stream):
            for t, cpu in self.entries[i]:
                t.data = cpu.to(self.device, non_blocking=True)
            self.events[i] = self.copy_stream.record_event()
        self.loaded[i] = True

    def _pre(self, i):
        def hook(module, args):
            self._load(i)
            cur = torch.cuda.current_stream(self.device)
            cur.wait_event(self.events[i])
            for t, _ in self.entries[i]:
                t.data.record_stream(cur)  # don't recycle this memory until compute is done with it
            # Throttle: the host would otherwise race ahead and stage every layer at once.
            prev = self.done[i - 1]
            if prev is not None:
                prev.synchronize()
            nxt = i + 1
            if nxt < len(self.layers):
                self._load(nxt)
            elif self.cyclic:
                self._load(0)
        return hook

    def _post(self, i):
        def hook(module, args, out):
            for t, cpu in self.entries[i]:
                t.data = cpu
            self.loaded[i] = False
            self.done[i] = torch.cuda.current_stream(self.device).record_event()
        return hook


def cpu_embedding(emb: nn.Embedding, device="cuda"):
    """Keep a (huge-vocab) embedding table in RAM; look up on CPU, ship only the result."""
    emb.to("cpu")
    emb.register_forward_pre_hook(lambda m, a: (a[0].to("cpu"),))
    emb.register_forward_hook(lambda m, a, out: out.to(device, non_blocking=True))
