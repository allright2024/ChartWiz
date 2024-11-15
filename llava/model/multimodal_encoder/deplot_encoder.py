import torch
import torch.nn as nn

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, Pix2StructConfig


class DeplotVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        # self.select_layer = args.mm_vision_select_layer
        # self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = Pix2StructConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        self.vision_tower = Pix2StructForConditionalGeneration.from_pretrained(self.vision_tower_name)
        self.image_processor = Pix2StructProcessor.from_pretrained(self.vision_tower_name)
        if self.vision_tower_name == "nuua/ko-deplot":
            self.vision_tower.load_state_dict(torch.load("/home/work/ko-deplot_finetuning/checkpoint/deplot_model_ver_20.10.19_korean_only_epoch3.bin"))
        elif self.vision_tower_name == "ybelkada/pix2struct-base":
            self.vision_tower.load_state_dict(torch.load("/home/work/ai-hub/pretrained_model/deplot/deplot_k.pt"))
        
        self.vision_tower = self.vision_tower.encoder
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.last_hidden_state
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list: # train
            inputs = {}
            patch = torch.cat([img['flattened_patches'] for img in images], dim=0) 
            inputs["flattened_patches"] = patch

            mask = torch.cat([img['attention_mask'] for img in images], dim=0) 
            inputs["attention_mask"] = mask
            image_forward_out = self.vision_tower(**inputs, output_hidden_states=False)
            image_features = image_forward_out.last_hidden_state
            
        else: # inference
            inputs = {}
            inputs['flattened_patches'] = images['flattened_patches'].squeeze(0).to(dtype=torch.bfloat16)
            inputs["attention_mask"] = images["attention_mask"].squeeze(0).to(dtype=torch.bfloat16)
            image_forward_outs = self.vision_tower(**inputs, output_hidden_states=False)
            image_features = image_forward_outs.last_hidden_state

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

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

