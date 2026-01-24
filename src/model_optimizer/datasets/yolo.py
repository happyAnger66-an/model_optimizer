from ultralytics.data.build import build_yolo_dataset, build_dataloader

def get_yolo_dataset(data_path, batch_size, mode):
    dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
    return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
        )