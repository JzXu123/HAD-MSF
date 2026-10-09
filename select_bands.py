"""SIFDR band selection used by the clean Pavia release."""
import numpy as np


def _pixel_matrix(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 3:
        return data.reshape(-1, data.shape[-1])
    if data.ndim == 2:
        return data
    raise ValueError("data must be HxWxB or NxB")


def SIFDR(pre: np.ndarray, bandnum: int, ridge: float = 1e-6) -> np.ndarray:
    """Select bands using the SIFDR spatial-spectral fusion criterion."""
    pixels = _pixel_matrix(pre)
    n_pixels, n_bands = pixels.shape
    bandnum = int(bandnum)
    if not 1 <= bandnum <= n_bands:
        raise ValueError(f"bandnum must be in [1, {n_bands}], got {bandnum}")
    if n_pixels == 0:
        raise ValueError("the image contains no pixels")

    covariance = (pixels.T @ pixels) / float(n_pixels)
    scale = max(float(np.trace(covariance)) / max(n_bands, 1), 1.0)
    inverse = np.linalg.pinv(covariance + ridge * scale * np.eye(n_bands))
    weights = np.empty((n_bands, n_pixels), dtype=np.float64)
    regularizer = ridge * scale * np.eye(n_bands)
    for start in range(0, n_pixels, 1000):
        stop = min(start + 1000, n_pixels)
        block = pixels[start:stop].T
        weights[:, start:stop] = np.linalg.solve(
            inverse @ block @ block.T + regularizer, inverse @ block
        )
        weights[:, start:stop] = np.abs(weights[:, start:stop])
    rho = weights.mean(axis=1)

    gram = pixels.T @ pixels
    squared_norm = np.diag(gram)
    distances = np.sqrt(np.maximum(
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * gram, 0.0
    ))
    distances = np.abs(np.tanh(distances / np.sqrt(n_pixels)))
    order = np.argsort(rho)[::-1]
    delta = np.zeros(n_bands, dtype=np.float64)
    delta[order[0]] = float(distances.max()) if distances.size else 1.0
    for rank in range(1, n_bands):
        delta[order[rank]] = min(distances[order[rank], order[:rank]])
    rho_range = float(rho.max() - rho.min())
    rho_norm = (rho - rho.min()) / rho_range if rho_range > 1e-12 else np.ones_like(rho)
    criterion = rho_norm * delta
    return np.asarray(np.argsort(criterion)[::-1][:bandnum], dtype=np.int64)
