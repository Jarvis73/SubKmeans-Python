import numpy as np
from pathlib import Path
from src.subkmeans.SubKmeans import SubKmeans
from src.utils import DataIO, DataNormalization, NormalizedMutualInformation
from src.K_means import k_means

def main_subspace_kmeans(ds, separator=";"):
    """ Subspace K-means
    """
    dataset = Path(f"./datasets/{ds}.dat")
    dataname = dataset.stem

    d, groundTruth = DataIO.loadCsvWithIntLabelsAsSeq(dataset, separator=separator)
    data = DataNormalization.standardizeData(d)

    nrOfClusters = np.unique(groundTruth).shape[0]

    handler = SubKmeans()
    result = handler.runWithReplicates(data, nrOfClusters, 10)

    nmi = NormalizedMutualInformation.forLabelSeq(groundTruth, result.labels)
    print(f"NMI: {nmi}")
    print(f"m: {result.m()}")

    transformedData = np.matmul(result.transformation().T, data[:, :, None]).squeeze()
    DataIO.writeClusters(Path("./result") / f"{dataname}_result.dat", transformedData, result.labels)


def main_kmeans(ds, separator=";"):
    """ K-means
    """
    dataset = Path(f"./datasets/{ds}.dat")

    d, groundTruth = DataIO.loadCsvWithIntLabelsAsSeq(dataset, separator=separator)
    data = DataNormalization.standardizeData(d)

    nrOfClusters = np.unique(groundTruth).shape[0]

    all_nmi = 0
    for _ in range(10):
        result_labels = k_means(data, nrOfClusters)
        nmi = NormalizedMutualInformation.forLabelSeq(groundTruth, result_labels)
        all_nmi += nmi
    
    print(f"NMI: {all_nmi / 10}")


if __name__ == "__main__":
    args = ["OliveOil", "  "]
    #main_kmeans(*args)
    main_subspace_kmeans(*args)
