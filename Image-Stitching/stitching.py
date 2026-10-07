"""Feature-based planar mosaics with PyTorch and Kornia.

The public function signatures follow the original CSE 473/573 submission.
All image geometry and compositing here operate on tensors, without file I/O.
"""
from typing import Dict
import torch
import kornia as K


class StitchingError(ValueError):
    """No defensible alignment or a numerically unsafe transform was found."""


MAX_CANVAS_PIXELS = 24_000_000
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.20
MIN_OVERLAP = 0.20


def _normalize_images(imgs):
    if not imgs:
        raise StitchingError("Provide at least one RGB image")
    keys = sorted(imgs)
    images = []
    device = imgs[keys[0]].device
    for key in keys:
        img = imgs[key]
        if img.ndim != 3 or img.shape[0] != 3 or min(img.shape[-2:]) < 2:
            raise StitchingError("Images must have shape (3, H, W), with H and W >= 2")
        if img.device != device:
            raise StitchingError("All images must use the same device")
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        elif img.is_floating_point():
            img = img.float()
            if not torch.isfinite(img).all() or img.min() < 0 or img.max() > 1:
                raise StitchingError("Float images must be finite and in [0, 1]")
        else:
            raise StitchingError("Use uint8 [0,255] or floating-point [0,1] images")
        images.append(img)
    return keys, images


def corners(height, width, homography):
    """Project pixel-center corners, rejecting a horizon crossing the image."""
    pts = homography.new_tensor([[0, width-1, width-1, 0],
                                 [0, 0, height-1, height-1], [1, 1, 1, 1]])
    transformed = homography @ pts
    denominator = transformed[2]
    if (not torch.isfinite(transformed).all() or denominator.abs().min() < 1e-6
            or not (torch.all(denominator > 0) or torch.all(denominator < 0))):
        raise StitchingError("Homography crosses infinity inside an image")
    return transformed[:2] / denominator.unsqueeze(0)


def _valid_homography(h):
    if h is None or h.shape != (3, 3) or not torch.isfinite(h).all():
        return False
    scale = h.abs().max()
    if scale < 1e-9:
        return False
    singular = torch.linalg.svdvals(h / scale)
    return bool(singular[-1] > 1e-8 and singular[0] / singular[-1] < 1e8)


def _canvas(images, transforms):
    all_corners = torch.cat([corners(*image.shape[-2:], h)
                             for image, h in zip(images, transforms)], dim=1)
    lower = torch.floor(all_corners.min(dim=1).values)
    upper = torch.ceil(all_corners.max(dim=1).values)
    width, height = (upper - lower + 1).to(torch.int64).tolist()
    if width < 1 or height < 1 or width * height > MAX_CANVAS_PIXELS:
        raise StitchingError("Estimated canvas is invalid or too large; check correspondences")
    translation = torch.eye(3, dtype=all_corners.dtype, device=all_corners.device)
    translation[0, 2], translation[1, 2] = -lower[0], -lower[1]
    return (height, width), [translation @ h for h in transforms]


def _support(image, homography, size):
    mask = torch.ones((1, 1, *image.shape[-2:]), device=image.device, dtype=image.dtype)
    return K.geometry.transform.warp_perspective(mask, homography.unsqueeze(0), size,
               mode="nearest", padding_mode="zeros", align_corners=True)[0]


def _pair_overlap(image_a, image_b, b_to_a):
    identity = torch.eye(3, device=image_a.device, dtype=image_a.dtype)
    size, transforms = _canvas([image_a, image_b], [identity, b_to_a])
    a = _support(image_a, transforms[0], size)
    b = _support(image_b, transforms[1], size)
    intersection = (a * b).sum()
    # Symmetric overlap criterion: fraction of the smaller projected support.
    return float(intersection / torch.minimum(a.sum(), b.sum()).clamp_min(1))


def _extract(images):
    sift = K.feature.SIFTFeature(num_features=800, rootsift=True).to(images[0].device).eval()
    features = []
    for image in images:
        gray = K.color.rgb_to_grayscale(image.unsqueeze(0))
        lafs, _, descriptors = sift(gray)
        features.append((lafs[0], descriptors[0]))
    return features


def matching(desc_a, desc_b, lafs_a, lafs_b):
    """Return H mapping image B coordinates INTO image A coordinates."""
    if min(len(desc_a), len(desc_b)) < 4:
        return None, None
    _, pairs = K.feature.DescriptorMatcher("smnn", 0.8)(desc_a, desc_b)
    if len(pairs) < MIN_INLIERS:
        return None, None
    points_a = K.feature.get_laf_center(lafs_a[pairs[:, 0]].unsqueeze(0))[0]
    points_b = K.feature.get_laf_center(lafs_b[pairs[:, 1]].unsqueeze(0))[0]
    estimator = K.geometry.ransac.RANSAC(model_type="homography", inl_th=3.0,
                                        batch_size=1024, max_iter=10)
    h, inliers = estimator(points_b, points_a)
    h = h.squeeze() if h is not None else None
    if inliers is None:
        return None, None
    count = int(inliers.sum())
    if count < MIN_INLIERS or count / len(pairs) < MIN_INLIER_RATIO or not _valid_homography(h):
        return None, inliers
    h = h / h[2, 2] if h[2, 2].abs() > 1e-8 else h / h.norm()
    return h, inliers


def _alignment_graph(images, features):
    n = len(images)
    adjacency = {i: {} for i in range(n)}
    overlap = torch.eye(n, dtype=torch.int64, device=images[0].device)
    for i in range(n):
        for j in range(i+1, n):
            lafs_i, desc_i = features[i]
            lafs_j, desc_j = features[j]
            h, inliers = matching(desc_i, desc_j, lafs_i, lafs_j)
            if h is None:
                continue
            try:
                if _pair_overlap(images[i], images[j], h) < MIN_OVERLAP:
                    continue
                inverse = torch.linalg.inv(h)
                if not _valid_homography(inverse):
                    continue
            except (StitchingError, RuntimeError):
                continue
            confidence = int(inliers.sum())
            adjacency[i][j] = (h, confidence)  # j -> i
            adjacency[j][i] = (inverse, confidence)  # i -> j
            overlap[i, j] = overlap[j, i] = 1
    return adjacency, overlap


def _largest_component(adjacency):
    unseen, components = set(adjacency), []
    while unseen:
        start = min(unseen)
        component, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node])
        unseen -= component
        components.append(component)
    return max(components, key=lambda c: (len(c), -min(c)))


def _compose_transforms(adjacency, selected, identity):
    reference = max(selected, key=lambda i: (len(adjacency[i]), -i))
    transforms = {reference: identity}
    queue = [reference]
    for node in queue:
        neighbors = sorted(adjacency[node], key=lambda j: (-adjacency[node][j][1], j))
        for neighbor in neighbors:
            if neighbor in selected and neighbor not in transforms:
                # neighbor -> node -> reference; left multiplication composes destinations.
                transforms[neighbor] = transforms[node] @ adjacency[node][neighbor][0]
                queue.append(neighbor)
    return transforms


def _feather(image):
    height, width = image.shape[-2:]
    x = torch.arange(width, device=image.device, dtype=image.dtype)
    y = torch.arange(height, device=image.device, dtype=image.dtype)
    dx = torch.minimum(x+1, width-x)
    dy = torch.minimum(y+1, height-y)
    return torch.minimum(dy[:, None], dx[None, :]).unsqueeze(0)


def _render(images, transforms, blend="feather"):
    """Warp explicit support masks, keeping legitimate black pixels valid."""
    if blend not in {"average", "feather"}:
        raise ValueError("Choose average or feather blending")
    size, translated = _canvas(images, transforms)
    accum = images[0].new_zeros((3, *size))
    weights = images[0].new_zeros((1, *size))
    for image, h in zip(images, translated):
        # Normalize bilinear interpolation by alpha to avoid dark boundary halos.
        support = _support(image, h, size)
        ones = image.new_ones((1, 1, *image.shape[-2:]))
        alpha = K.geometry.transform.warp_perspective(ones, h.unsqueeze(0), size,
                   align_corners=True)[0]
        warped = K.geometry.transform.warp_perspective(image.unsqueeze(0), h.unsqueeze(0), size,
                   align_corners=True)[0] / alpha.clamp_min(1e-8)
        source_weight = _feather(image) if blend == "feather" else ones[0]
        weight = K.geometry.transform.warp_perspective(source_weight.unsqueeze(0), h.unsqueeze(0), size,
                     align_corners=True)[0] * support
        accum += warped * weight
        weights += weight
    return (accum / weights.clamp_min(1e-8)).clamp(0, 1).mul(255).round().to(torch.uint8).cpu()


@torch.inference_mode()
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """Align two images and feather their mosaic; not reliable foreground removal."""
    _, images = _normalize_images(imgs)
    if len(images) != 2:
        raise StitchingError("stitch_background expects exactly two images")
    features = _extract(images)
    h, _ = matching(features[0][1], features[1][1], features[0][0], features[1][0])
    if h is None:
        raise StitchingError("Not enough reliable correspondences to align the two images")
    if _pair_overlap(images[0], images[1], h) < MIN_OVERLAP:
        raise StitchingError("The two images have insufficient projected overlap")
    identity = torch.eye(3, device=images[0].device, dtype=images[0].dtype)
    return _render(images, [identity, h])


@torch.inference_mode()
def panorama(imgs: Dict[str, torch.Tensor]):
    """Stitch the largest connected group; return RGB uint8 and binary N x N overlap."""
    _, images = _normalize_images(imgs)
    identity = torch.eye(3, device=images[0].device, dtype=images[0].dtype)
    if len(images) == 1:
        return _render(images, [identity]), torch.ones((1, 1), dtype=torch.int64)
    graph, overlap = _alignment_graph(images, _extract(images))
    selected = _largest_component(graph)
    if len(selected) < 2:
        raise StitchingError("No connected overlapping image group was found")
    transforms = _compose_transforms(graph, selected, identity)
    indices = sorted(selected)
    return _render([images[i] for i in indices], [transforms[i] for i in indices]), overlap.cpu()
