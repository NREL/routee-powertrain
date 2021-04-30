from unittest import TestCase

from os import remove
from pathlib import Path

from powertrain.validation.feature_visualization import visualize_features

from tests.mock_resources import mock_model


def _clean_temp_files(filepath: Path):
    """
    removes the given directory and all temp files within

    :param filepath: directory to be deleted
    """
    for dir in filepath.glob('*'):
        for f in dir.glob('*'):
            try:
                remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        dir.rmdir()
    filepath.rmdir()


class TestVisualizeFeatures(TestCase):

    def test_successful_run(self):
        """
        test to verify that the predictions returned are the correct length, contain the tests for each of the features
        in the model, and plots are saved to the correct location with the correct naming scheme
        """
        model = mock_model()
        model_name = model.metadata.model_description
        estimator_name = model.metadata.estimator_name
        feature_ranges = {
            'gpsspeed': {
                'max': 80,
                'min': 0,
                'default': 40
            },
            'grade': {
                'max': 0.20,
                'min': -0.20,
                'default': 0
            }
        }
        # temp directory for holding temporary results
        output_filepath = 'tmp/'
        Path(output_filepath).mkdir(parents=True, exist_ok=True)

        # run the function with the mock data
        predictions = visualize_features(model=model,
                                         feature_ranges=feature_ranges,
                                         num_links=15,
                                         output_path=output_filepath)
        # tests to check the predictions
        try:
            self.assertEqual(list(predictions.keys()), ['gpsspeed', 'grade'], 'should have tested both grade and '
                                                                              'gpsspeed')
            self.assertEqual(len(predictions['gpsspeed']), 15, 'should have made predictions for 15 links testing '
                                                               'gpsspeed')
            self.assertEqual(len(predictions['grade']), 15, 'should have made predictions for 15 links testing grade')

            # tests for saving plots and naming convention
            self.assertTrue(Path.exists(Path(output_filepath).joinpath(f'{model_name}/{estimator_name}_[grade].png')),
                            'should save grade plot as png')
            self.assertTrue(Path.exists(Path(output_filepath).joinpath(f'{model_name}/{estimator_name}_[gpsspeed].png')),
                            'should save gpsspeed plot as png')

        except AssertionError as error:
            # clean up temp files
            raise AssertionError(error)
        finally:
            _clean_temp_files(Path(output_filepath))

    def test_missing_feature(self):
        """
        test to verify that a KeyError is thrown when the config is missing a required feature
        """
        model = mock_model()
        feature_ranges = {
            'gpsspeed': {
                'max': 80,
                'min': 0,
                'default': 40
            }
        }

        # temp directory for holding temporary results
        output_filepath = Path('tmp/')
        output_filepath.mkdir(parents=True, exist_ok=True)

        try:
            with self.assertRaises(KeyError):
                visualize_features(model=model,
                                   feature_ranges=feature_ranges,
                                   num_links=15,
                                   output_path=output_filepath)

        except AssertionError as error:
            # clean up temp files
            raise AssertionError(error)

        finally:
            _clean_temp_files(output_filepath)
