#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "../graphics/Renderer.h"
#include "../graphics/RendererConfig.h"
#include "../core/DataProcessor.h"

namespace nb = nanobind;

NB_MODULE(qsplot_engine, m) {
    m.doc() = "QsPlot: High-performance Visualization for Quantitative Finance";

    // ---------------------------
    // RendererConfig Binding
    // ---------------------------
    nb::class_<RendererConfig>(m, "RendererConfig")
        .def(nb::init<>())
        .def_rw("window_width", &RendererConfig::windowWidth)
        .def_rw("window_height", &RendererConfig::windowHeight)
        .def_rw("vsync", &RendererConfig::vsync)
        .def_rw("point_scale", &RendererConfig::pointScale)
        .def_rw("global_alpha", &RendererConfig::globalAlpha)
        .def_rw("color_mode", &RendererConfig::colorMode);

    // ---------------------------
    // Renderer Binding
    // ---------------------------
    nb::class_<Renderer>(m, "Renderer")
        .def(nb::init<>())
        .def(nb::init<const RendererConfig&>())
        .def("start", &Renderer::start, "Start the rendering thread")
        .def("stop", &Renderer::stop, "Stop the rendering thread")
        .def("set_points", [](Renderer& self, 
                              nb::ndarray<float, nb::ndim<2>, nb::c_contig> positions, 
                              nb::ndarray<float, nb::ndim<1>, nb::c_contig> values) {
            
            // Check dimensions
            if (positions.shape(1) != 3) {
                throw std::runtime_error("Positions must be N x 3");
            }
            if (positions.shape(0) != values.shape(0)) {
                throw std::runtime_error("Positions and Values must have same row count");
            }

            self.setPoints(positions.data(), values.data(), positions.shape(0));
        }, "Upload N x 3 positions and N x 1 values to the renderer")
        
        .def("set_target_points", [](Renderer& self, 
                              nb::ndarray<float, nb::ndim<2>, nb::c_contig> positions, 
                              nb::ndarray<float, nb::ndim<1>, nb::c_contig> values) {
            // Check dimensions
            if (positions.shape(1) != 3) throw std::runtime_error("Positions must be N x 3");
            if (positions.shape(0) != values.shape(0)) throw std::runtime_error("Positions and Values must have same row count");
            
            self.setTargetPoints(positions.data(), values.data(), positions.shape(0));
        }, "Upload next frame data (Morph Target) (N x 3, N x 1)")
        
        .def("get_selected_id", &Renderer::getSelectedID, "Get the index of the currently selected point (-1 if none)")
        
        .def("set_points_raw", [](Renderer& self, 
                              nb::ndarray<float, nb::ndim<2>, nb::c_contig> positions, 
                              nb::ndarray<float, nb::ndim<1>, nb::c_contig> values) {
                                  
            if (positions.shape(1) != 3) throw std::runtime_error("Positions must be N x 3");
            if (positions.shape(0) != values.shape(0)) throw std::runtime_error("Rows mismatch");
            
            self.setPointsRaw(positions.data(), values.data(), positions.shape(0));
        }, "Directly upload 3D coordinates (bypassing internal logic). Positions must be scaled by user.")
        
        .def("set_tickers", &Renderer::setTickers, "Set ticker labels for each point")
        .def("get_selected_ticker", &Renderer::getSelectedTicker, "Get the ticker of the currently selected point")
        .def("save_screenshot", &Renderer::saveScreenshot, "Save a screenshot to the specified path (PPM format)")
        .def("set_dimension_labels", &Renderer::setDimensionLabels, 
             "Set labels for dimensions (color, x, y, z) to display in UI");

    // ---------------------------
    // DataProcessor Binding
    // ---------------------------
    nb::class_<DataProcessor>(m, "DataProcessor")
        .def(nb::init<>())
        .def("load_data", &DataProcessor::loadData, "Load raw data matrix")
        .def("compute_pca", &DataProcessor::computePCA, nb::arg("target_dims") = 3, "Reduce to 3D using PCA")
        .def("get_explained_variance_ratio", &DataProcessor::getExplainedVarianceRatio, "Get variance ratio per component")
        .def("extract_feature", &DataProcessor::extractFeature, "Get column vector");
}
