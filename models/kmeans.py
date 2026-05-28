# Third-party libraries
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import graycomatrix, graycoprops
from sklearn.base import BaseEstimator, TransformerMixin


class K_Means(BaseEstimator, TransformerMixin):
    """
    Custom transformer applying K-Means clustering on image pixels to extract features.

    The K-Means model is fitted on a pooled sample of pixels collected from all input images,
    allowing the model to learn global color clusters shared across the dataset.

    This transformer extracts:
    - color cluster histograms (in LAB space)
    - intra-cluster statistics (mean and standard deviation)
    - global color statistics
    - contrast features
    - edge-based features (Sobel)
    - texture features using GLCM
    - radial labels histograms

    It is designed to be used within a scikit-learn pipeline for image classification tasks.

    Parameters
    ----------
    k : int, default=5
        Number of clusters.
    sample_size : int, default=300
        Number of pixels sampled per image for fitting K-Means.
    n_init : int, default=10
        Number of K-Means initializations.
    image_size : int, default=32
        Image size (images are assumed to be square).
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self, k=5, sample_size=300, n_init=10, img_size=(128, 128), random_state=42
    ):
        self.dtype = np.float64
        self.k = k
        self.sample_size = sample_size
        self.img_size = img_size
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, images, y=None):
        """
        Fit the K-Means model on a pooled sample of pixels extracted from all images.

        Parameters
        ----------
        images : np.ndarray of shape (n_samples, n_pixels)
            Flattened input images.
        y : None, optional
            Ignored.

        Returns
        -------
        self : K_Means
            Fitted transformer.
        """

        images = np.array(images)

        all_pixels = []

        rng = np.random.default_rng(self.random_state)

        for img in images:

            img = img.reshape(*self.img_size, 3)

            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            p = img_lab.reshape(-1, 3)

            if len(p) == 0:
                continue

            n_samples = min(self.sample_size, p.shape[0])

            idx = rng.choice(p.shape[0], size=n_samples, replace=False)
            all_pixels.append(p[idx])

        X_sample = np.vstack(all_pixels).astype(np.float64)

        self.kmeans_ = MiniBatchKMeans(
            n_clusters=self.k,
            batch_size=1000,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.kmeans_.fit(X_sample)

        # Cluster order by luminosity (L)
        self.order_ = np.argsort(self.kmeans_.cluster_centers_[:, 0])

        return self

    def transform(self, images):
        """
        Transform images into feature vectors using the fitted K-Means model.

        Each image is processed independently. Pixel-level cluster assignments are
        computed using the globally fitted K-Means model, and aggregated into
        descriptive features.

        Parameters
        ----------
        images : ndarray of shape (n_samples, n_pixels)
            Flattened input images.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Transformed feature matrix.
        """

        features = []

        for img in images:
            feat = []

            img = img.reshape(*self.img_size, 3)

            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            pixels = img_lab.reshape(-1, 3).astype(np.float64)
            labels = self.kmeans_.predict(pixels)

            mapping = np.zeros_like(self.order_)
            mapping[self.order_] = np.arange(len(self.order_))

            labels = mapping[labels]

            # Cluster histograms
            hist, _ = np.histogram(labels, bins=np.arange(self.k + 1))
            hist = hist / (hist.sum() + 1e-8)
            hist = np.sqrt(hist)

            feat.extend(hist)

            # Intra cluster mean and std
            for k in range(self.k):
                mask = labels == k

                if mask.any():
                    cluster_k = pixels[mask]

                    feat.extend(cluster_k.mean(axis=0))
                    feat.extend(cluster_k.std(axis=0))

                else:
                    feat.extend([0, 0, 0])
                    feat.extend([0, 0, 0])

            # Mean (L, A, B)
            feat.extend(pixels.mean(axis=0))

            # std (L, A, B)
            feat.extend(pixels.std(axis=0))

            # Contours (Sobel)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

            magnitude = np.hypot(sobelx, sobely)
            feat.extend([magnitude.mean(), magnitude.std()])

            # GLCM (Gray Level Co-occurrence Matrix)
            glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
            props = ["contrast", "homogeneity", "energy"]

            feat.extend([graycoprops(glcm, p)[0, 0] for p in props])

            # Radial histograms
            cluster_map = labels.reshape(self.img_size)

            radial_features = radial_kmeans_histograms(
                cluster_map=cluster_map, n_clusters=self.k
            )

            feat.extend(radial_features)

            features.append(np.asarray(feat, dtype=np.float32))

        return np.array(features)

    def get_feature_names(self):
        """
        Get the names of the extracted features.

        Returns
        -------
        feature_names : list of str
            Names corresponding to each feature in the transformed output.
        """

        feature_names = []

        for k in range(self.k):
            feature_names.append(f"Hist {k+1}")

        for i in range(self.k):
            for c in ["L", "A", "B"]:
                feature_names.append(f"Intra mean {i+1} {c}")

        for i in range(self.k):
            for c in ["L", "A", "B"]:
                feature_names.append(f"Intra std {i+1} {c}")

        for c in ["L", "A", "B"]:
            feature_names.append(f"Global mean {c}")

        for c in ["L", "A", "B"]:
            feature_names.append(f"Global std {c}")

        for stat in ["mean", "std"]:
            feature_names.append(f"Sobel {stat}")

        for tex in ["contrast", "homogeneity", "energy"]:
            feature_names.append(f"GLCM {tex}")

        for radii in ["r1", "r2", "r3", "r4"]:
            for k in range(self.k):
                feature_names.append(f"{radii} hist {k+1}")

        return feature_names

    def get_cluster_centers(self):
        """
        Get cluster centers ordered by luminance.

        Returns
        -------
        centers : ndarray of shape (n_clusters, 3)
            Cluster centers in LAB color space, sorted by brightness (L channel).
        """

        return self.kmeans_.cluster_centers_[self.order_]


def radial_kmeans_histograms(
    cluster_map,
    n_clusters,
    radii=(0.1, 0.2, 0.3, 0.4),
):
    """
    Compute radial histograms of KMeans clusters.

    Parameters
    ----------
    cluster_map : np.ndarray (H, W)
        Cluster index per pixel.

    n_clusters : int
        Number of KMeans clusters.

    radii : tuple
        Relative radii defining concentric rings.

    Returns
    -------
    np.ndarray
        Concatenated radial histograms.
    """

    h, w = cluster_map.shape

    cy = h / 2
    cx = w / 2

    y, x = np.ogrid[:h, :w]
    dist = np.hypot(x - cx, y - cy)

    max_radius = np.hypot(cx, cy)
    dist = dist / max_radius

    features = []
    prev_r = 0.0

    for r in radii:

        mask = (dist >= prev_r) & (dist < r)
        labels = cluster_map[mask]

        hist, _ = np.histogram(
            labels,
            bins=n_clusters,
            range=(0, n_clusters),
        )

        hist = hist / (hist.sum() + 1e-8)
        hist = np.sqrt(hist)

        features.extend(hist)

        prev_r = r

    return np.array(features)
