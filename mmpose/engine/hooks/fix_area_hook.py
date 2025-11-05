# hooks/fix_area_hook.py

from mmengine.hooks import Hook
from mmengine.runner import Runner
from tqdm import tqdm

from mmpose.registry import HOOKS

@HOOKS.register_module()
class FixAreaListHook(Hook):
    """Hook to fix annotation `area` field if it is a list instead of float."""
    priority = 'LOW'  # 可改为 'VERY_LOW' 等更低优先级

    def before_train(self, runner: Runner):
        # This runs before each train epoch; adjust as needed (can also use before_val_epoch)
        self._fix_area_field(runner)

    def before_val(self, runner: Runner):
        self._fix_area_field(runner)

    # def before_test_epoch(self, runner):
    #     print("✅ FixAreaListHook.before_test_epoch triggered")
    #     self._fix_area_field(runner)

    def before_test(self, runner):
        print("✅ FixAreaListHook.before_test triggered")
        self._fix_area_field(runner)

    def _fix_area_field(self, runner: Runner):
        # Iterate dataset annotations (if available via runner)
        if hasattr(runner, 'train_dataloader'):
            ds = runner.train_dataloader.dataset
            for data in tqdm(ds, desc="Fixing training dataset area fields", total=len(ds)):
                data['data_samples'].raw_ann_info['area'] = data['data_samples'].raw_ann_info['area'][0]
        # Also fix validation dataset if present
        if hasattr(runner, 'val_dataloader'):
            ds = runner.val_dataloader.dataset
            for data in tqdm(ds, desc="Fixing val dataset area fields", total=len(ds)):
                data['data_samples'].raw_ann_info['area'] = data['data_samples'].raw_ann_info['area'][0]

        if hasattr(runner, 'test_dataloader'):
            ds = runner.test_dataloader.dataset
            for data in tqdm(ds, desc="Fixing test dataset area fields", total=len(ds)):
                data['data_samples'].raw_ann_info['area'] = data['data_samples'].raw_ann_info['area'][0]
