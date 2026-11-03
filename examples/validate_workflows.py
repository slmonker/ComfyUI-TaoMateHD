from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KNOWN_TYPES = {
    "UNETLoader", "LoraLoaderModelOnly", "MiniMaxH3SigmaShift", "CLIPLoader",
    "VAELoader", "MiniMaxH3ImageToVideo", "TaoMateHDProgressiveSampler",
    "LTXVSeparateAVLatent", "VAEDecode", "VAEDecodeAudio", "CreateVideo",
    "SaveVideo", "MarkdownNote",
}


def validate(ui_path: Path, api_path: Path):
    ui = json.loads(ui_path.read_text(encoding="utf-8"))
    api = json.loads(api_path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in ui["nodes"]}
    assert ui["last_node_id"] == max(nodes)
    assert ui["last_link_id"] == len(ui["links"])
    for node in nodes.values():
        assert node["type"] in KNOWN_TYPES, node["type"]
    link_by_id = {link[0]: link for link in ui["links"]}
    for link in ui["links"]:
        link_id, src, src_slot, dst, dst_slot, type_name = link
        assert src in nodes and dst in nodes
        assert link_id in nodes[src]["outputs"][src_slot]["links"]
        assert nodes[dst]["inputs"][dst_slot]["link"] == link_id
        assert nodes[src]["outputs"][src_slot]["type"] == type_name
        assert nodes[dst]["inputs"][dst_slot]["type"] == type_name
    for key, node in api.items():
        assert node["class_type"] in KNOWN_TYPES, node["class_type"]
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in api, (key, value)
    print(f"{ui_path.name}: nodes={len(nodes)} links={len(link_by_id)} api_nodes={len(api)} OK")


def main():
    for stem in ("taomate_hd_t2va_5s_480p", "taomate_hd_t2va_5s_768p"):
        validate(HERE / f"{stem}.json", HERE / f"{stem}_api.json")


if __name__ == "__main__":
    main()
