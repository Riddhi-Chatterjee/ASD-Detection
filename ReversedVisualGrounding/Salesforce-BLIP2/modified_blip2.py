import torch
import torch.nn as nn
import copy  # For deep copying
from blip2_models.blip2 import Blip2Base  # Import Blip2Base class
from transformers import T5TokenizerFast, T5Config, T5ForConditionalGeneration

class ModifiedBLIP2(Blip2Base):
    def __init__(
        self,
        pretrained_blip2,  # Pretrained BLIP-2 model passed as the first parameter
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32", # Warning: This isn't used as we load blip2's pretrained VIT
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        max_txt_len=32,
        prompt="",
    ):
        super().__init__()

        # Deep copy vision encoders and layer norms for independent use
        self.image_encoder = pretrained_blip2.visual_encoder.to(torch.float32)
        self.mask_encoder = self.image_encoder
        self.ln_vision_image = pretrained_blip2.ln_vision
        self.ln_vision_mask = self.ln_vision_image
        self.freeze_module(self.ln_vision_image)
        self.freeze_module(self.ln_vision_mask)

        # Freeze encoders if required
        if freeze_vit:
            self.freeze_module(self.image_encoder)
            self.freeze_module(self.mask_encoder)

        # Deep copy Q-Formers for independent use (preloaded weights)
        self.q_former_image = pretrained_blip2.Qformer
        self.freeze_module(self.q_former_image)
        self.q_former_mask = copy.deepcopy(pretrained_blip2.Qformer)
        # self.freeze_module(self.q_former_mask)
        
        state_dict = pretrained_blip2.state_dict()

        # Extract the pretrained query tokens from the state_dict
        pretrained_query_tokens = state_dict["query_tokens"].clone()

        # Create independent query tokens for image and mask as trainable parameters
        self.query_tokens_image = nn.Parameter(pretrained_query_tokens.clone(), requires_grad=False)
        self.query_tokens_mask = nn.Parameter(pretrained_query_tokens.clone(), requires_grad=True)

        # Initialize T5 tokenizer and T5 model
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)
        self.freeze_module(self.t5_model)  # Freeze T5 model

        # Linear projection layer to align Q-Former output with T5 input space
        self.t5_proj = nn.Linear(self.q_former_image.config.hidden_size, t5_config.d_model)

        # Cross-attention mechanism with batch_first=True
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        # Store max text length and prompt
        self.max_txt_len = max_txt_len
        self.prompt = prompt

    def freeze_module(self, module):
        """Helper function to freeze module parameters."""
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, image, mask, text_input=None, text_output=None, max_seq_len=32):
        # Step 1: Extract visual features and apply layer normalization
        img_features = self.ln_vision_image(self.image_encoder(image))
        mask_features = self.ln_vision_mask(self.mask_encoder(mask))

        # Step 2: Generate queries using the independent Q-Formers
        query_tokens_img = self.query_tokens_image.expand(img_features.size(0), -1, -1)
        query_output_img = self.q_former_image.bert(
            query_embeds=query_tokens_img,
            encoder_hidden_states=img_features,
            encoder_attention_mask=torch.ones(img_features.size()[:-1], dtype=torch.long).to(image.device),
            return_dict=True,
        ).last_hidden_state

        query_tokens_mask = self.query_tokens_mask.expand(mask_features.size(0), -1, -1)
        query_output_mask = self.q_former_mask.bert(
            query_embeds=query_tokens_mask,
            encoder_hidden_states=mask_features,
            encoder_attention_mask=torch.ones(mask_features.size()[:-1], dtype=torch.long).to(mask.device),
            return_dict=True,
        ).last_hidden_state

        # Step 3: Apply cross-attention between image and mask queries
        q_final, _ = self.cross_attention(query_output_img, query_output_mask, query_output_mask)

        # Step 4: Project queries to T5 input space
        inputs_t5 = self.t5_proj(q_final)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", truncation=True, max_length=self.max_txt_len, return_tensors="pt"
        ).to(image.device)
        output_tokens = self.t5_tokenizer(
            text_output, padding="longest", truncation=True, max_length=self.max_txt_len, return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        # Step 5: Forward pass through T5 model for loss calculation
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        
        return outputs
