import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clone_module(module, number_of_copies):
    """
    Create independent copies of a module.

    deepcopy is required here. Repeating the same module object would make
    all Transformer layers share parameters.
    """
    return nn.ModuleList(
        [
            copy.deepcopy(module)
            for _ in range(number_of_copies)
        ]
    )


def _get_activation_function(name):
    if name == "relu":
        return F.relu

    if name == "gelu":
        return F.gelu

    if name == "glu":
        return F.glu

    raise ValueError(
        f"Unsupported activation function: {name}"
    )


def _add_position_embedding(tensor, position):
    """
    Add positional information when it is available.
    """
    if position is None:
        return tensor

    return tensor + position


class ACTTransformerEncoderLayer(nn.Module):
    """
    One DETR-style Transformer encoder layer.

    Args:
        d_model:
            Transformer embedding dimension.

        nhead:
            Number of attention heads.

        dim_feedforward:
            Hidden dimension of the feed-forward network.

        dropout:
            Dropout probability.

        activation:
            "relu", "gelu", or "glu".

        normalize_before:
            False reproduces the original DETR/ACT post-norm structure.
            True enables pre-norm.
    """

    def __init__(
        self,
        d_model=256,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(
            d_model,
            dim_feedforward,
        )

        self.linear2 = nn.Linear(
            dim_feedforward,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = _get_activation_function(
            activation
        )

        self.normalize_before = normalize_before

    def _forward_post_norm(
        self,
        source,
        position=None,
        attention_mask=None,
        padding_mask=None,
    ):
        """
        Original DETR-style post-norm encoder layer.
        """
        query = _add_position_embedding(
            source,
            position,
        )

        key = _add_position_embedding(
            source,
            position,
        )

        value = source

        attention_output = self.self_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]

        source = (
            source
            + self.dropout1(attention_output)
        )

        source = self.norm1(source)

        feedforward_output = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(source)
                )
            )
        )

        source = (
            source
            + self.dropout2(feedforward_output)
        )

        source = self.norm2(source)

        return source

    def _forward_pre_norm(
        self,
        source,
        position=None,
        attention_mask=None,
        padding_mask=None,
    ):
        """
        Pre-norm encoder layer.

        This mode is usually more numerically stable for deep Transformers,
        but normalize_before=False is closer to the original ACT code.
        """
        normalized_source = self.norm1(source)

        query = _add_position_embedding(
            normalized_source,
            position,
        )

        key = _add_position_embedding(
            normalized_source,
            position,
        )

        value = normalized_source

        attention_output = self.self_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]

        source = (
            source
            + self.dropout1(attention_output)
        )

        normalized_source = self.norm2(source)

        feedforward_output = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(normalized_source)
                )
            )
        )

        source = (
            source
            + self.dropout2(feedforward_output)
        )

        return source

    def forward(
        self,
        source,
        position=None,
        attention_mask=None,
        padding_mask=None,
    ):
        """
        Args:
            source:
                [B, S, D] source token content.

            position:
                None or [B, S, D] source position embeddings.

            attention_mask:
                Optional self-attention mask.

            padding_mask:
                None or [B, S].
                True means that the source token is padding.

        Returns:
            Encoded source with shape [B, S, D].
        """
        if self.normalize_before:
            return self._forward_pre_norm(
                source=source,
                position=position,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
            )

        return self._forward_post_norm(
            source=source,
            position=position,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
        )


class ACTTransformerEncoder(nn.Module):
    """
    Stack of DETR-style Transformer encoder layers.
    """

    def __init__(
        self,
        encoder_layer,
        number_of_layers,
        normalization=None,
    ):
        super().__init__()

        self.layers = _clone_module(
            encoder_layer,
            number_of_layers,
        )

        self.number_of_layers = number_of_layers
        self.normalization = normalization

    def forward(
        self,
        source,
        position=None,
        attention_mask=None,
        padding_mask=None,
    ):
        """
        Args:
            source:
                [B, S, D]

            position:
                None or [B, S, D]

            attention_mask:
                Optional encoder self-attention mask.

            padding_mask:
                None or [B, S], True means padding.

        Returns:
            Memory tensor with shape [B, S, D].
        """
        output = source

        for layer in self.layers:
            output = layer(
                source=output,
                position=position,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
            )

        if self.normalization is not None:
            output = self.normalization(output)

        return output


class ACTTransformerDecoderLayer(nn.Module):
    """
    One DETR-style Transformer decoder layer.

    Each layer contains:

        1. Self-attention among action queries.
        2. Cross-attention from action queries to encoded observation memory.
        3. Feed-forward network.

    Action-query embeddings are passed as target_position. They are not used
    as the initial target content.
    """

    def __init__(
        self,
        d_model=256,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(
            d_model,
            dim_feedforward,
        )

        self.linear2 = nn.Linear(
            dim_feedforward,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = _get_activation_function(
            activation
        )

        self.normalize_before = normalize_before

    def _forward_post_norm(
        self,
        target,
        memory,
        target_position=None,
        memory_position=None,
        target_attention_mask=None,
        memory_attention_mask=None,
        target_padding_mask=None,
        memory_padding_mask=None,
    ):
        """
        Original DETR-style post-norm decoder layer.
        """

        # 1. Self-attention between action queries.
        query = _add_position_embedding(
            target,
            target_position,
        )

        key = _add_position_embedding(
            target,
            target_position,
        )

        self_attention_output = (
            self.self_attention(
                query=query,
                key=key,
                value=target,
                attn_mask=target_attention_mask,
                key_padding_mask=target_padding_mask,
                need_weights=False,
            )[0]
        )

        target = (
            target
            + self.dropout1(
                self_attention_output
            )
        )

        target = self.norm1(target)

        # 2. Cross-attention from action queries to observation memory.
        query = _add_position_embedding(
            target,
            target_position,
        )

        key = _add_position_embedding(
            memory,
            memory_position,
        )

        cross_attention_output = (
            self.cross_attention(
                query=query,
                key=key,
                value=memory,
                attn_mask=memory_attention_mask,
                key_padding_mask=memory_padding_mask,
                need_weights=False,
            )[0]
        )

        target = (
            target
            + self.dropout2(
                cross_attention_output
            )
        )

        target = self.norm2(target)

        # 3. Feed-forward network.
        feedforward_output = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(target)
                )
            )
        )

        target = (
            target
            + self.dropout3(
                feedforward_output
            )
        )

        target = self.norm3(target)

        return target

    def _forward_pre_norm(
        self,
        target,
        memory,
        target_position=None,
        memory_position=None,
        target_attention_mask=None,
        memory_attention_mask=None,
        target_padding_mask=None,
        memory_padding_mask=None,
    ):
        """
        Pre-norm decoder layer.
        """

        # 1. Self-attention between action queries.
        normalized_target = self.norm1(target)

        query = _add_position_embedding(
            normalized_target,
            target_position,
        )

        key = _add_position_embedding(
            normalized_target,
            target_position,
        )

        self_attention_output = (
            self.self_attention(
                query=query,
                key=key,
                value=normalized_target,
                attn_mask=target_attention_mask,
                key_padding_mask=target_padding_mask,
                need_weights=False,
            )[0]
        )

        target = (
            target
            + self.dropout1(
                self_attention_output
            )
        )

        # 2. Cross-attention.
        normalized_target = self.norm2(target)

        query = _add_position_embedding(
            normalized_target,
            target_position,
        )

        key = _add_position_embedding(
            memory,
            memory_position,
        )

        cross_attention_output = (
            self.cross_attention(
                query=query,
                key=key,
                value=memory,
                attn_mask=memory_attention_mask,
                key_padding_mask=memory_padding_mask,
                need_weights=False,
            )[0]
        )

        target = (
            target
            + self.dropout2(
                cross_attention_output
            )
        )

        # 3. Feed-forward network.
        normalized_target = self.norm3(target)

        feedforward_output = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(normalized_target)
                )
            )
        )

        target = (
            target
            + self.dropout3(
                feedforward_output
            )
        )

        return target

    def forward(
        self,
        target,
        memory,
        target_position=None,
        memory_position=None,
        target_attention_mask=None,
        memory_attention_mask=None,
        target_padding_mask=None,
        memory_padding_mask=None,
    ):
        """
        Args:
            target:
                [B, K, D] decoder target content.
                ACT initializes this tensor to zeros.

            memory:
                [B, S, D] output of the policy Transformer encoder.

            target_position:
                [B, K, D] learned action-query embeddings.

            memory_position:
                [B, S, D] source position embeddings.

            target_attention_mask:
                Optional decoder self-attention mask.

            memory_attention_mask:
                Optional decoder cross-attention mask.

            target_padding_mask:
                None or [B, K].

            memory_padding_mask:
                None or [B, S].

        Returns:
            Decoded action features with shape [B, K, D].
        """
        if self.normalize_before:
            return self._forward_pre_norm(
                target=target,
                memory=memory,
                target_position=target_position,
                memory_position=memory_position,
                target_attention_mask=target_attention_mask,
                memory_attention_mask=memory_attention_mask,
                target_padding_mask=target_padding_mask,
                memory_padding_mask=memory_padding_mask,
            )

        return self._forward_post_norm(
            target=target,
            memory=memory,
            target_position=target_position,
            memory_position=memory_position,
            target_attention_mask=target_attention_mask,
            memory_attention_mask=memory_attention_mask,
            target_padding_mask=target_padding_mask,
            memory_padding_mask=memory_padding_mask,
        )


class ACTTransformerDecoder(nn.Module):
    """
    Stack of DETR-style Transformer decoder layers.
    """

    def __init__(
        self,
        decoder_layer,
        number_of_layers,
        normalization=None,
        return_intermediate=False,
    ):
        super().__init__()

        self.layers = _clone_module(
            decoder_layer,
            number_of_layers,
        )

        self.number_of_layers = number_of_layers
        self.normalization = normalization
        self.return_intermediate = (
            return_intermediate
        )

    def forward(
        self,
        target,
        memory,
        target_position=None,
        memory_position=None,
        target_attention_mask=None,
        memory_attention_mask=None,
        target_padding_mask=None,
        memory_padding_mask=None,
    ):
        """
        Returns:
            When return_intermediate=False:
                [B, K, D]

            When return_intermediate=True:
                [L, B, K, D], where L is the number
                of decoder layers.
        """
        output = target
        intermediate_outputs = []

        for layer in self.layers:
            output = layer(
                target=output,
                memory=memory,
                target_position=target_position,
                memory_position=memory_position,
                target_attention_mask=target_attention_mask,
                memory_attention_mask=memory_attention_mask,
                target_padding_mask=target_padding_mask,
                memory_padding_mask=memory_padding_mask,
            )

            if self.return_intermediate:
                if self.normalization is not None:
                    normalized_output = (
                        self.normalization(output)
                    )
                else:
                    normalized_output = output

                intermediate_outputs.append(
                    normalized_output
                )

        if self.normalization is not None:
            output = self.normalization(output)

        if self.return_intermediate:
            # Replace the final intermediate output with the
            # fully normalized final decoder output.
            intermediate_outputs[-1] = output

            return torch.stack(
                intermediate_outputs,
                dim=0,
            )

        return output


class ACTTransformer(nn.Module):
    """
    Complete DETR-style encoder-decoder Transformer for ACT.

    Data flow:

        source content + source positions
                        |
                        v
                Transformer Encoder
                        |
                      memory
                        |
        zero target + action query positions
                        |
                        v
                Transformer Decoder
                        |
                        v
                 action features

    This module does not contain the visual backbone, CVAE latent encoder,
    action head, or padding head. Those belong to ACTPolicy.
    """

    def __init__(
        self,
        d_model=256,
        nhead=8,
        number_of_encoder_layers=4,
        number_of_decoder_layers=7,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_decoder=False,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead, "
                f"but received d_model={d_model}, "
                f"nhead={nhead}."
            )

        if number_of_encoder_layers <= 0:
            raise ValueError(
                "number_of_encoder_layers must be positive."
            )

        if number_of_decoder_layers <= 0:
            raise ValueError(
                "number_of_decoder_layers must be positive."
            )

        encoder_layer = ACTTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )

        # Original DETR adds a final encoder norm only for pre-norm.
        encoder_normalization = (
            nn.LayerNorm(d_model)
            if normalize_before
            else None
        )

        self.encoder = ACTTransformerEncoder(
            encoder_layer=encoder_layer,
            number_of_layers=(
                number_of_encoder_layers
            ),
            normalization=encoder_normalization,
        )

        decoder_layer = ACTTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )

        # DETR always applies a final normalization to decoder output.
        decoder_normalization = nn.LayerNorm(
            d_model
        )

        self.decoder = ACTTransformerDecoder(
            decoder_layer=decoder_layer,
            number_of_layers=(
                number_of_decoder_layers
            ),
            normalization=decoder_normalization,
            return_intermediate=(
                return_intermediate_decoder
            ),
        )

        self.d_model = d_model
        self.nhead = nhead
        self.return_intermediate_decoder = (
            return_intermediate_decoder
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Match DETR's Xavier initialization for matrix parameters.
        """
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def _validate_inputs(
        source,
        source_position,
        query_position,
        source_padding_mask,
    ):
        if source.ndim != 3:
            raise ValueError(
                "source must have shape [B, S, D], "
                f"but received {tuple(source.shape)}."
            )

        if source_position is not None:
            if source_position.shape != source.shape:
                raise ValueError(
                    "source_position must have the same "
                    "shape as source. Received "
                    f"{tuple(source_position.shape)} and "
                    f"{tuple(source.shape)}."
                )

        if query_position.ndim != 3:
            raise ValueError(
                "query_position must have shape [B, K, D], "
                f"but received {tuple(query_position.shape)}."
            )

        if query_position.shape[0] != source.shape[0]:
            raise ValueError(
                "source and query_position batch sizes "
                "must match."
            )

        if query_position.shape[2] != source.shape[2]:
            raise ValueError(
                "source and query_position embedding "
                "dimensions must match."
            )

        if source_padding_mask is not None:
            expected_shape = (
                source.shape[0],
                source.shape[1],
            )

            if tuple(source_padding_mask.shape) != expected_shape:
                raise ValueError(
                    "source_padding_mask must have shape "
                    f"{expected_shape}, but received "
                    f"{tuple(source_padding_mask.shape)}."
                )

            if source_padding_mask.dtype != torch.bool:
                raise TypeError(
                    "source_padding_mask must have "
                    "torch.bool dtype."
                )

    def forward(
        self,
        source,
        source_position,
        query_position,
        source_padding_mask=None,
    ):
        """
        Args:
            source:
                Source content with shape [B, S, D].

                ACT source order will normally be:

                    [latent, proprioception, visual tokens...]

            source_position:
                Position embeddings with shape [B, S, D].

                ACT position order will normally be:

                    [latent position,
                     proprioception position,
                     visual positions...]

            query_position:
                Learned action-query positions with shape [B, K, D].

            source_padding_mask:
                Optional bool tensor with shape [B, S].
                True means that a source token is padding.

                In standard ACT visual inference, this will normally be
                None because latent, proprioception, and image features
                are all valid.

        Returns:
            If return_intermediate_decoder=False:
                Action-query features with shape [B, K, D].

            If return_intermediate_decoder=True:
                Decoder outputs with shape [L, B, K, D].
        """
        self._validate_inputs(
            source=source,
            source_position=source_position,
            query_position=query_position,
            source_padding_mask=source_padding_mask,
        )

        memory = self.encoder(
            source=source,
            position=source_position,
            padding_mask=source_padding_mask,
        )

        # Official ACT/DETR uses zero decoder content.
        # Learned action queries are injected separately as positions.
        target = torch.zeros_like(
            query_position
        )

        output = self.decoder(
            target=target,
            memory=memory,
            target_position=query_position,
            memory_position=source_position,
            memory_padding_mask=(
                source_padding_mask
            ),
        )

        return output