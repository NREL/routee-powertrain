from unittest import TestCase

from os import remove
from pathlib import Path

from powertrain.validation.feature_visualization import visualize_features

from tests.mock_resources import mock_model


def _clean_temp_files(filepath: Path):
    for f in filepath.glob('*.png'):
        try:
            remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    filepath.rmdir()


class TestVisualizeFeatures(TestCase):

    def test_successful_run(self):
        model = mock_model()
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
        output_filepath = Path('tmp/')
        output_filepath.mkdir(parents=True, exist_ok=True)

        # run the function with the mock data
        predictions = visualize_features(model=model,
                                         feature_ranges=feature_ranges,
                                         output_filepath=output_filepath,
                                         num_links=15)
        # tests to check the predictions
        self.assertEqual(list(predictions.keys()), ['gpsspeed', 'grade'], 'should have tested both grade and gpsspeed')
        self.assertEqual(len(predictions['gpsspeed']), 15, 'should have made predictions for 15 links testing gpsspeed')
        self.assertEqual(len(predictions['grade']), 15, 'should have made predictions for 15 links testing grade')

        # tests for saving plots and naming convention
        self.assertTrue(Path.exists(output_filepath.joinpath(f'{model.metadata.model_description}_grade.png')),
                        'should save grade plot as png')
        self.assertTrue(Path.exists(output_filepath.joinpath(f'{model.metadata.model_description}_gpsspeed.png')),
                        'should save gpsspeed plot as png')

        # clean up temp files
        _clean_temp_files(output_filepath)

    def test_missing_feature(self):
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

        with self.assertRaises(KeyError):
            visualize_features(model=model,
                               feature_ranges=feature_ranges,
                               output_filepath=output_filepath,
                               num_links=15)

        _clean_temp_files(output_filepath)
