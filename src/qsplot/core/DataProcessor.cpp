#include "DataProcessor.h"
#include <iostream>

DataProcessor::DataProcessor() : m_hasData(false) {
}

DataProcessor::~DataProcessor() {
}

void DataProcessor::loadData(const Eigen::Ref<const Eigen::MatrixXd>& data) {
    m_rawData = data;
    m_hasData = true;
    std::cout << "[DataProcessor] Loaded data: " << m_rawData.rows() << " x " << m_rawData.cols() << std::endl;
}

Eigen::MatrixXd DataProcessor::computePCA(int targetDims) {
    if (!m_hasData || m_rawData.rows() == 0) {
        return Eigen::MatrixXd();
    }

    // 1. Center and Standardize the data (Z-Score)
    // Important for financial data with different units (e.g., Price vs Volume)
    Eigen::RowVectorXd means = m_rawData.colwise().mean();
    Eigen::MatrixXd centered = m_rawData.rowwise() - means;
    
    // Calculate StdDev for each column
    Eigen::RowVectorXd stdDevs = (centered.array().square().colwise().sum() / double(m_rawData.rows() - 1)).sqrt();
    
    // Avoid division by zero
    for(int i=0; i<stdDevs.size(); ++i) {
        if(stdDevs[i] < 1e-9) stdDevs[i] = 1.0;
    }
    
    Eigen::MatrixXd standardized = centered.array().rowwise() / stdDevs.array();

    // 2. Compute Covariance Matrix of Standardized Data (Correlation Matrix)
    Eigen::MatrixXd cov = (standardized.adjoint() * standardized) / double(m_rawData.rows() - 1);

    // 3. Eigendecomposition (SelfAdjoint for symmetric matrix)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    
    // 4. Sort eigenvectors
    Eigen::MatrixXd eigenVectors = solver.eigenvectors().rightCols(targetDims);
    
    // Sort eigenvalues for variance explanation (ascending order from solver)
    Eigen::VectorXd eigenValues = solver.eigenvalues();
    
    // Provide feedback on Variance Explained (optional logging)
    double totalVariance = eigenValues.sum();
    if (totalVariance > 1e-9) {
        // Take top N eigenvalues (which are at the end)
        m_explainedVariance = eigenValues.tail(targetDims).reverse() / totalVariance;
        
        double explained = eigenValues.tail(targetDims).sum();
        std::cout << "[DataProcessor] PCA Variance Explained: " << (explained / totalVariance) * 100.0 << "%" << std::endl;
    } else {
        m_explainedVariance = Eigen::VectorXd::Zero(targetDims);
    }

    // Reverse columns to have largest first
    Eigen::MatrixXd sortedVectors = eigenVectors.rowwise().reverse();

    // 5. Project standardized data
    return standardized * sortedVectors;
}

Eigen::VectorXd DataProcessor::getExplainedVarianceRatio() const {
    return m_explainedVariance;
}

Eigen::VectorXd DataProcessor::extractFeature(int colIndex) {
    if (!m_hasData || colIndex < 0 || colIndex >= m_rawData.cols()) {
        return Eigen::VectorXd();
    }
    return m_rawData.col(colIndex);
}

size_t DataProcessor::getSampleCount() const {
    if (!m_hasData) return 0;
    return m_rawData.rows();
}
