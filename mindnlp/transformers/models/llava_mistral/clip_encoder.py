import mindspore as ms
import mindspore.nn as nn

from mindnlp.transformers.models.clip import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Cell):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(
            args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name,
                                                             # local_files_only=True,
                                                             # cache_dir='/mindnlp_models'
                                                             )

    def load_model(self):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(
                self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name,
                                                                  # local_files_only=True,
                                                                  # cache_dir='/mindnlp_models'
                                                                  )
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name,
                                                            # local_files_only=True,
                                                            # cache_dir='/mindnlp_models'
                                                            )

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(
                f'Unexpected select feature: {self.select_feature}')
        return image_features

    def construct(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.astype(
                    self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(
                    image_forward_out).astype(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.astype(self.dtype), output_hidden_states=True)
            image_features = self.feature_select(
                image_forward_outs).astype(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return ms.ops.zeros(1, self.hidden_size, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
