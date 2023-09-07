from unittest import TestCase

from os import remove
from pathlib import Path

from nrel.routee.powertrain.validation.feature_visualization import (
    visualize_features,
    contour_plot,
)

from tests.mock_resources import mock_model


def _clean_temp_files_multi_directory(filepath: str):
    """
    removes the given directory, sub directories, and all temp files within

    :param filepath: directory to be deleted
    """
    for dir in Path(filepath).glob("*"):
        for f in dir.glob("*"):
            try:
                remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        dir.rmdir()
    Path(filepath).rmdir()


def _clean_temp_files_single_directory(filepath: str):
    """
    removes the given directory and all temp files within

    :param filepath: directory to be deleted
    """
    for f in Path(filepath).glob("*"):
        try:
            remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    Path(filepath).rmdir()


class TestVisualizeFeatures(TestCase):
    def test_visualize_features_successful_run(self):
        """
        test to verify that the predictions returned are the correct length, contain the tests for each of the features
        in the model, and plots are saved to the correct location with the correct naming scheme
        """
        model = mock_model()
        model_name = model.metadata.model_description
        estimator_name = model.metadata.estimator_name
        feature_ranges = {
            "speed": {"max": 80, "min": 0, "steps": 40},
            "grade": {"max": 0.5, "min": -0.5, "steps": 20},
        }
        # temp directory for holding temporary results
        output_filepath = "tmp/"
        Path(output_filepath).mkdir(parents=True, exist_ok=True)

        # run the function with the mock data
        predictions = visualize_features(
            model=model, feature_ranges=feature_ranges, output_path=output_filepath
        )
        # tests to check the predictions
        try:
            self.assertEqual(
                list(predictions.keys()),
                ["speed", "grade"],
                "should have tested both grade and " "speed",
            )
            self.assertEqual(
                len(predictions["speed"]),
                40,
                "should have made predictions for 15 links testing " "speed",
            )
            self.assertEqual(
                len(predictions["grade"]),
                20,
                "should have made predictions for 15 links testing grade",
            )

            # tests for saving plots and naming convention
            self.assertTrue(
                Path.exists(
                    Path(output_filepath).joinpath(
                        f"{model_name}/{estimator_name}_[grade].png"
                    )
                ),
                "should save grade plot as png",
            )
            self.assertTrue(
                Path.exists(
                    Path(output_filepath).joinpath(
                        f"{model_name}/{estimator_name}_[speed].png"
                    )
                ),
                "should save speed plot as png",
            )

        except AssertionError as error:
            # clean up temp files
            raise AssertionError(error)
        finally:
            _clean_temp_files_multi_directory(output_filepath)

    def test_visualize_features_missing_feature(self):
        """
        test to verify that a KeyError is thrown when the config is missing a required feature
        """
        model = mock_model()
        feature_ranges = {"speed": {"max": 80, "min": 0, "steps": 40}}

        # temp directory for holding temporary results
        output_filepath = Path("tmp/")
        output_filepath.mkdir(parents=True, exist_ok=True)

        try:
            with self.assertRaises(KeyError):
                visualize_features(
                    model=model,
                    feature_ranges=feature_ranges,
                    output_path=output_filepath,
                )

        except AssertionError as error:
            # clean up temp files
            raise AssertionError(error)

        finally:
            _clean_temp_files_multi_directory(output_filepath)

    def test_contour_plot_successful_run(self):
        """
        test to verify a contour plot is successfully saved to the tmp directory
        """
        model = mock_model()
        model_name = model.metadata.model_description
        feature_ranges = {
            "speed": {"max": 80, "min": 0, "steps": 40},
            "grade": {"max": 0.5, "min": -0.5, "steps": 20},
        }
        # temp directory for holding temporary results
        output_filepath = "tmp/"
        Path(output_filepath).mkdir(parents=True, exist_ok=True)

        # run the function with the mock data
        contour_plot(
            model=model,
            x_feature="speed",
            y_feature="grade",
            feature_ranges=feature_ranges,
            output_path=output_filepath,
        )
        # tests to check the predictions
        try:
            # tests for saving plots and naming convention
            self.assertTrue(
                Path.exists(
                    Path(output_filepath).joinpath(f"{model_name}_[speed_grade].png")
                ),
                "should save contour plot as png",
            )
        except AssertionError as error:
            # clean up temp files
            raise AssertionError(error)
        finally:
            _clean_temp_files_single_directory(output_filepath)

    def test_contour_plot_incompatible_feature(self):
        """
        test to verify a KeyError is thrown if the x/y test features are not supported by the model
        """
        model = mock_model()
        feature_ranges = {
            "speed": {"max": 80, "min": 0, "steps": 40},
            "grade": {"max": 0.5, "min": -0.5, "steps": 20},
        }
        # temp directory for holding temporary results
        output_filepath = "tmp/"
        Path(output_filepath).mkdir(parents=True, exist_ok=True)

        try:
            with self.assertRaises(KeyError):
                contour_plot(
                    model=model,
                    x_feature="warp_speed",
                    y_feature="grade",
                    feature_ranges=feature_ranges,
                    output_path=output_filepath,
                )

        except AssertionError as error:
            # clean up temp files
            raise AssertionError(error)

        finally:
            _clean_temp_files_single_directory(output_filepath)

    def test_contour_plot_missing_feature(self):
        """
        test to verify that a KeyError is thrown if a required feature is missing from the feature ranges
        """
        model = mock_model()
        feature_ranges = {"speed": {"max": 80, "min": 0, "steps": 40}}
        # temp directory for holding temporary results
        output_filepath = "tmp/"
        Path(output_filepath).mkdir(parents=True, exist_ok=True)

        try:
            with self.assertRaises(KeyError):
                contour_plot(
                    model=model,
                    x_feature="speed",
                    y_feature="grade",
                    feature_ranges=feature_ranges,
                    output_path=output_filepath,
                )

        except AssertionError as error:
            # clean up temp files
            raise AssertionError(error)

        finally:
            _clean_temp_files_single_directory(output_filepath)
