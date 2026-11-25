from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from cs400.analysis import analyze_reports
from cs400.config import Candidate
from cs400.evaluation import CORRUPTIONS, robustness_metrics
from cs400.storage import write_json


class AnalysisBoundaryTests(unittest.TestCase):
    def report(self):
        scores={(c,s):.6 for c in CORRUPTIONS for s in range(1,6)}
        return dict(phase='post_search',training_identity='one-trained-model',manifest_id='same-data',epoch=60,
                    candidate=asdict(Candidate()), metrics=robustness_metrics(.8,scores),
                    conditions=[dict(corruption=c,severity=s,accuracy=a) for (c,s),a in scores.items()])

    def test_insufficient_observations_do_not_fit_predictor(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'evaluation.json';write_json(path,self.report())
            result=analyze_reports([path,path],Path(temp)/'analysis.json')
            self.assertEqual(result['status'],'descriptive_only')
            self.assertEqual(result['distinct_configurations'],1)
            self.assertEqual(result['trained_models'],1)
            self.assertNotIn('held_out_permutation_importance',result)

    def test_incomplete_corruptions_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            report=self.report();report['conditions'].pop()
            path=Path(temp)/'evaluation.json';write_json(path,report)
            with self.assertRaises(ValueError):analyze_reports([path],Path(temp)/'analysis.json')

    def test_repeated_configuration_with_different_weights_keeps_both_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            first=self.report();second=self.report()
            first['artifact_id']='weights-one';second['artifact_id']='weights-two'
            second['metrics']['clean_accuracy']=.81
            second['metrics']['accuracy_drop_percentage_points']=21.0
            paths=[Path(temp)/'one.json',Path(temp)/'two.json']
            for path,value in zip(paths,[first,second]):write_json(path,value)
            result=analyze_reports(paths,Path(temp)/'analysis.json')
            self.assertEqual(result['trained_models'],2)
            self.assertEqual(result['distinct_configurations'],1)

    def test_inconsistent_aggregate_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            report=self.report();report['metrics']['mean_corruption_error']=.1
            path=Path(temp)/'evaluation.json';write_json(path,report)
            with self.assertRaises(ValueError):analyze_reports([path],Path(temp)/'analysis.json')


if __name__=='__main__':
    unittest.main()
