#pragma once

#include <Eigen/Dense>
#include <vector>

class DataProcessor {
public:
    DataProcessor();
    ~DataProcessor();

    /**
     * @brief Load raw data matrix (Rows = Samples, Cols = Dimensions)
     * 
     * @param data Eigen Matrix reference
     */
    void loadData(const Eigen::Ref<const Eigen::MatrixXd>& data);

    /**
     * @brief Perform PCA reduction to N dimensions
     * 
     * @param targetDims Number of dimensions to maximize variance for (usually 3 for XYZ)
     * @return Eigen::MatrixXd Reduced matrix (Rows x targetDims)
     */
    Eigen::MatrixXd computePCA(int targetDims = 3);
    
    /**
     * @brief Get the explained variance ratio of the last PCA computation
     */
    Eigen::VectorXd getExplainedVarianceRatio() const;

    /**
     * @brief Extract a specific column as a vector (e.g., for Color mapping)
     * 
     * @param colIndex Column index in the original raw data
     * @return Eigen::VectorXd Column vector
     */
    Eigen::VectorXd extractFeature(int colIndex);

    /**
     * @brief Get the number of samples
     */
    size_t getSampleCount() const;

private:
    Eigen::MatrixXd m_rawData;
    bool m_hasData;
    Eigen::VectorXd m_explainedVariance;
};
