from .metric import Metric


class YoloMetric(Metric):
    def __init__(self, result):
        super().__init__(result)

    def diff(self, values1, values2):
        return [((v2 - v1) / v1) * 100.0 if v1 != 0 else v2 for v1, v2 in zip(values1, values2)]

    def compare(self, other):
        print(self.metric_header)
        metrics = self.result
        other_metrics = other.get_result()
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(metrics.keys)
        diff_pf = "%22s" + "%11i" * 2 + "%10.3g%%" * len(metrics.keys)

        for i, c in enumerate(metrics.ap_class_index):
            metric_values = metrics.class_result(i)
            other_metric_values = other_metrics.class_result(i)
            diff_values = self.diff(metric_values, other_metric_values)
            print()
            print('--------------------------------')
            print(self.metric_header)
            print(pf
                  % (
                      metrics.names[c],
                      metrics.nt_per_image[c],
                      metrics.nt_per_class[c],
                      *metrics.class_result(i),
                  )
                  )
            print(pf % (other_metrics.names[c], other_metrics.nt_per_image[c],
                  other_metrics.nt_per_class[c], *other_metrics.class_result(i)))
            print(diff_pf % (other_metrics.names[c], other_metrics.nt_per_image[c],
                  other_metrics.nt_per_class[c], *diff_values))
            print('--------------------------------')
            print()

    @property
    def metric_header(self):
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def print(self):
        print(self.metric_header)
        metrics = self.result
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(metrics.keys)
        print(pf % ("all", 0, metrics.nt_per_class.sum(), *metrics.mean_results()))
        for i, c in enumerate(metrics.ap_class_index):
            print(pf
                  % (
                      metrics.names[c],
                      metrics.nt_per_image[c],
                      metrics.nt_per_class[c],
                      *metrics.class_result(i),
                  )
                  )
