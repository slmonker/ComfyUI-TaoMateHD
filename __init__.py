from .nodes import TaoMateHDProgressiveSampler

NODE_CLASS_MAPPINGS = {
    "TaoMateHDProgressiveSampler": TaoMateHDProgressiveSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TaoMateHDProgressiveSampler": "TaoMate HD Progressive Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
