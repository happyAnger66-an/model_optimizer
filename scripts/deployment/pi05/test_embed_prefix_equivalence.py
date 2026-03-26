import argparse
import math
import time

import torch

from openpi.training import config as _config
from openpi.policies import policy_config


def _old_embed_prefix(model, images, img_masks, lang_tokens, lang_masks):
    embs = []
    pad_masks = []
    att_masks = []

    for img, img_mask in zip(images, img_masks, strict=True):
        image_outputs = model.paligemma_with_expert.paligemma.model.vision_tower(img)
        selected_image_feature = image_outputs.last_hidden_state
        img_emb = model.paligemma_with_expert.paligemma.model.multi_modal_projector(selected_image_feature)
        img_emb = img_emb / math.sqrt(float(model.paligemma_with_expert.paligemma.config.text_config.hidden_size))
        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        att_masks += [0] * num_img_embs

    lang_emb = model.paligemma_with_expert.paligemma.language_model.embed_tokens(lang_tokens)
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb = lang_emb * math.sqrt(float(lang_emb_dim))
    embs.append(lang_emb)
    pad_masks.append(lang_masks)

    num_lang_embs = lang_emb.shape[1]
    att_masks += [0] * num_lang_embs

    out_embs = torch.cat(embs, dim=1)
    out_pad = torch.cat(pad_masks, dim=1)
    att_t = torch.tensor(att_masks, dtype=torch.bool, device=out_pad.device)
    bsize = out_pad.shape[0]
    att_t = att_t[None, :].expand(bsize, att_t.shape[0])
    return out_embs, out_pad, att_t


def _new_embed_prefix(model, images, img_masks, lang_tokens, lang_masks):
    batched_images = torch.stack(images, dim=1)
    bsize, num_views = batched_images.shape[:2]
    flat_images = batched_images.reshape(
        bsize * num_views,
        batched_images.shape[2],
        batched_images.shape[3],
        batched_images.shape[4],
    )

    image_outputs = model.paligemma_with_expert.paligemma.model.vision_tower(flat_images)
    selected_image_feature = image_outputs.last_hidden_state
    flat_img_emb = model.paligemma_with_expert.paligemma.model.multi_modal_projector(selected_image_feature)
    flat_img_emb = flat_img_emb / math.sqrt(float(model.paligemma_with_expert.paligemma.config.text_config.hidden_size))

    num_img_embs = flat_img_emb.shape[1]
    img_emb = flat_img_emb.reshape(bsize, num_views, num_img_embs, flat_img_emb.shape[-1])
    img_emb = img_emb.reshape(bsize, num_views * num_img_embs, flat_img_emb.shape[-1])

    stacked_img_masks = torch.stack(img_masks, dim=1)
    img_pad_masks = stacked_img_masks[:, :, None].expand(bsize, num_views, num_img_embs)
    img_pad_masks = img_pad_masks.reshape(bsize, num_views * num_img_embs)

    lang_emb = model.paligemma_with_expert.paligemma.language_model.embed_tokens(lang_tokens)
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb = lang_emb * math.sqrt(float(lang_emb_dim))

    out_embs = torch.cat([img_emb, lang_emb], dim=1)
    out_pad = torch.cat([img_pad_masks, lang_masks], dim=1)

    att_len = out_pad.shape[1]
    att_t = torch.zeros((bsize, att_len), dtype=torch.bool, device=out_pad.device)
    return out_embs, out_pad, att_t


def _benchmark(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser("Compare old/new embed_prefix implementations")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, default="pi05_libero")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--token_len", type=int, default=200)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This script is intended to run on CUDA device")

    config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(config, args.model_path)
    model = policy._model.eval().to(device)

    image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    image_dtype = model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
    images = [
        torch.randn(args.batch_size, 3, 224, 224, device=device, dtype=image_dtype)
        for _ in image_keys
    ]
    img_masks = [torch.ones(args.batch_size, device=device, dtype=torch.bool) for _ in image_keys]
    lang_tokens = torch.randint(0, 2048, (args.batch_size, args.token_len), device=device, dtype=torch.int64)
    lang_masks = torch.ones(args.batch_size, args.token_len, device=device, dtype=torch.bool)

    with torch.inference_mode():
        old_out = _old_embed_prefix(model, images, img_masks, lang_tokens, lang_masks)
        new_out = _new_embed_prefix(model, images, img_masks, lang_tokens, lang_masks)

    embs_diff = (old_out[0] - new_out[0]).abs()
    pad_equal = torch.equal(old_out[1], new_out[1])
    att_equal = torch.equal(old_out[2], new_out[2])

    print(f"embs max abs diff: {embs_diff.max().item():.8f}")
    print(f"embs mean abs diff: {embs_diff.mean().item():.8f}")
    print(f"pad masks equal: {pad_equal}")
    print(f"att masks equal: {att_equal}")

    with torch.inference_mode():
        old_ms = _benchmark(
            lambda: _old_embed_prefix(model, images, img_masks, lang_tokens, lang_masks),
            iters=args.iters,
            warmup=args.warmup,
        )
        new_ms = _benchmark(
            lambda: _new_embed_prefix(model, images, img_masks, lang_tokens, lang_masks),
            iters=args.iters,
            warmup=args.warmup,
        )
    print(f"old avg latency: {old_ms:.3f} ms")
    print(f"new avg latency: {new_ms:.3f} ms")
    print(f"speedup: {old_ms / new_ms:.3f}x")


if __name__ == "__main__":
    main()

