import datasets
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm

def evaluate_for_subjects(model, data_gen, subjects, user_level_data, hyperparams, hyperparams_features, model_type,
                          alert_threshold=0.5, rolling_window=0, print_output=False):
    erisk_metricst2 = EriskScoresT1T2()
    threshold = alert_threshold

    for subject in tqdm(set(subjects), total=len(subjects)):
        try:
            user_level_data_subject = {subject: user_level_data[subject]}
        except:
            continue
        true_label = user_level_data_subject[subject]['label']

        if print_output: print(subject, "Label", true_label)

        if model_type == "HAN" or model_type == "HSAN":
            predictions = model.predict(data_gen(user_level_data_subject, {'test': [subject]},
                                                 set_type='test', hyperparams_features=hyperparams_features,
                                                 seq_len=hyperparams['maxlen'],
                                                 batch_size=hyperparams['batch_size'],  # on all data at once
                                                 max_posts_per_user=None,
                                                 posts_per_group=hyperparams['posts_per_group'],
                                                 post_groups_per_user=None, compute_liwc=True,
                                                 shuffle=False), verbose=0)
        elif model_type == "HAN_BERT" or model_type == "Con_HAN":
            predictions = model.predict(data_gen(user_level_data_subject, {'test': [subject]},
                                                 set_type='test', hyperparams_features=hyperparams_features,
                                                 seq_len=hyperparams['maxlen'],
                                                 batch_size=hyperparams['batch_size'],  # on all data at once
                                                 model_type = model_type,
                                                 max_posts_per_user=None,
                                                 posts_per_group=hyperparams['posts_per_group'],
                                                 post_groups_per_user=None, compute_liwc=True,
                                                 shuffle=False), verbose=0)
        else:
            raise Exception("Unknown type!")

        predictions = [p[0] for p in predictions]
        if rolling_window:
            rolling_predictions = []
            # The first predictions will be copied
            rolling_predictions[:rolling_window - 1] = predictions[:rolling_window - 1]
            # rolling average over predictions
            rolling_predictions.extend(np.convolve(predictions, np.ones(rolling_window), 'valid') / rolling_window)
            predictions = rolling_predictions
        for prediction in predictions:
            model_prediction = int(prediction >= threshold)

            if print_output: print("Prediction: ", prediction, model_prediction)

            erisk_metricst2.add(prediction=model_prediction, reference={'label': true_label, 'user': subject})
            if print_output: print('prediction and reference', model_prediction, {'label': true_label, 'user': subject})
    return erisk_metricst2.compute(posts_per_datapoint=hyperparams['posts_per_group'])

def _penalty(k, p=0.0078):
    return -1 + 2 / (1 + np.exp(-p * (k - 1)))


def _lc(k, o):
    return 1 - (1 / (1 + np.exp(k - o)))


class EriskScoresT1T2(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="Erisk Metrics",
            citation="",
            inputs_description="predictions, references (contains label and user), n_posts",
            features=datasets.Features({
                'predictions': datasets.Value('int64'),
                'references': datasets.DatasetDict({'label': datasets.Value('int64'),
                                                    'user': datasets.Value('string')
                                                    })
            }),
            codebase_urls=[],
            reference_urls=[],
        )

    def _latency(self, predictions, references, posts_per_datapoint):
        assert (len(predictions) == len(references)), \
            "Number of predictions not equal to number of references: %d vs %d." % (len(predictions), len(references))

        predictions_per_user = {}
        labels_per_user = {}
        for i in range(len(predictions)):
            u = references[i]['user']
            l = references[i]['label']
            p = predictions[i]
            if u not in predictions_per_user:
                predictions_per_user[u] = []
            predictions_per_user[u].append(p)
            if u in labels_per_user:
                assert (labels_per_user[u] == l), "Inconsistent labels for same user: %s" % u
            else:
                labels_per_user[u] = l
        users = list(labels_per_user.keys())
        latencies = []
        for u in users:
            # Latency only relevant for true positives
            if labels_per_user[u] != 1 or sum(predictions_per_user[u]) == 0:
                continue
            i = 0
            p = predictions_per_user[u][i]
            # Minimum latency has to be the number of posts used for the first prediction,
            # assuming we predict 0s by default (before the model generated any predictions)
            latency = posts_per_datapoint
            while (p != 1) and (i < len(predictions_per_user[u])):
                latency += posts_per_datapoint
                p = predictions_per_user[u][i]
                i += 1

            latencies.append(latency)
        median_penalty = _penalty(np.median(latencies))
        print(latencies, median_penalty)
        return median_penalty

    def _erde(self, predictions, references, posts_per_datapoint, o):

        c_fp = 0.1
        c_fn = 1
        c_tp = 0.5

        assert (len(predictions) == len(references)), \
            "Number of predictions not equal to number of references: %d vs %d." % (len(predictions), len(references))

        predictions_per_user = {}
        labels_per_user = {}
        for i in range(len(predictions)):
            u = references[i]['user']
            l = references[i]['label']
            p = predictions[i]
            if u not in predictions_per_user:
                predictions_per_user[u] = []
            predictions_per_user[u].append(p)
            if u in labels_per_user:
                assert (labels_per_user[u] == l), "Inconsistent labels for same user: %s" % u
            else:
                labels_per_user[u] = l
        users = list(labels_per_user.keys())
        penalties = []
        for u in users:
            #False positive:
            if labels_per_user[u] != 1 and sum(predictions_per_user[u]) > 0:
                penalties.append(c_fp)

            #False negative:
            if labels_per_user[u] == 1 and sum(predictions_per_user[u]) == 0:
                penalties.append(c_fn)

            #True negative:
            if labels_per_user[u] != 1 and sum(predictions_per_user[u]) == 0:
                penalties.append(0)

            # True positive
            if labels_per_user[u] == 1 and sum(predictions_per_user[u]) > 0:
                i = 0
                p = predictions_per_user[u][i]
                latency = posts_per_datapoint
                while (p != 1) and (i < len(predictions_per_user[u])):
                    latency += posts_per_datapoint
                    p = predictions_per_user[u][i]
                    i += 1
                penalties.append(_lc(latency,o)*c_tp)

        assert len(penalties) == len(users)
        erde = np.mean(penalties)
        return erde

    def _compute(self, predictions, references, posts_per_datapoint):

        assert (len(predictions) == len(references)), \
            "Number of predictions not equal to number of references: %d vs %d." % (len(predictions), len(references))
        predictions_per_user = {}
        labels_per_user = {}
        for i in range(len(predictions)):
            u = references[i]['user']
            l = references[i]['label']
            p = predictions[i]
            if u not in predictions_per_user:
                predictions_per_user[u] = p
            # User-level prediction is 1 if any 1 was emitted, otherwise it's 0
            predictions_per_user[u] = (p or predictions_per_user[u])
            if u in labels_per_user:
                assert (labels_per_user[u] == l), "Inconsistent labels for same user: %s" % u
            else:
                labels_per_user[u] = l
        users = list(labels_per_user.keys())
        y_true = [labels_per_user[u] for u in users]
        y_pred = [predictions_per_user[u] for u in users]
        penalty_score = self._latency(predictions, references, posts_per_datapoint)
        return {"precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
                "latency_f1": f1_score(y_true, y_pred) * (1 - penalty_score),
                "erde5": self._erde(predictions, references, posts_per_datapoint, 5),
                "erde50": self._erde(predictions, references, posts_per_datapoint, 50)}