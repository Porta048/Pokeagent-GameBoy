"""
Test per Vision Encoder (architettura ispirata a DeepSeek-VL2).

Verifica:
1. PixelShuffleAdaptor: Compressione corretta 2×2
2. MultiHeadLatentAttention: KV cache compression
3. VisionPPO: Forward pass corretto
4. PPONetworkGroup: Funzionamento corretto
"""
import pytest
import torch


class TestPixelShuffleAdaptor:
    """Test per PixelShuffleAdaptor."""

    def test_pixel_shuffle_down_shape(self):
        """Verifica che pixel shuffle riduca correttamente le dimensioni spaziali."""
        from src.vision_encoder import PixelShuffleAdaptor

        adaptor = PixelShuffleAdaptor(
            input_channels=64,
            hidden_dim=256,
            output_dim=256,
            shuffle_factor=2
        )

        # Input: (B, C, H, W) = (1, 64, 16, 18)
        x = torch.randn(1, 64, 16, 18)

        # Dopo pixel shuffle: (B, C*4, H/2, W/2)
        shuffled = adaptor.pixel_shuffle_down(x)

        assert shuffled.shape == (1, 256, 8, 9), \
            f"Shape errato: {shuffled.shape}, expected (1, 256, 8, 9)"

    def test_pixel_shuffle_output_shape(self):
        """Verifica output finale dell'adaptor."""
        from src.vision_encoder import PixelShuffleAdaptor

        adaptor = PixelShuffleAdaptor(
            input_channels=64,
            hidden_dim=256,
            output_dim=256,
            shuffle_factor=2
        )

        # Input tipico dopo CNN: (B, 64, 15, 17)
        x = torch.randn(1, 64, 15, 17)
        output = adaptor(x)

        # Output: (B, N, D) dove N = (H_padded/2) * (W_padded/2)
        assert output.dim() == 3, f"Output deve essere 3D, got {output.dim()}"
        assert output.shape[0] == 1, "Batch size errato"
        assert output.shape[2] == 256, "Embedding dim errato"

    def test_pixel_shuffle_preserves_information(self):
        """Verifica che pixel shuffle non perda informazione (è invertibile)."""
        from src.vision_encoder import PixelShuffleAdaptor

        adaptor = PixelShuffleAdaptor(
            input_channels=64,
            hidden_dim=256,
            output_dim=256,
            shuffle_factor=2
        )

        x = torch.randn(2, 64, 16, 16)
        shuffled = adaptor.pixel_shuffle_down(x)

        # Verifica che tutti i valori siano presenti (no perdita)
        assert shuffled.numel() == x.numel(), "Pixel shuffle ha perso elementi"


class TestMultiHeadLatentAttention:
    """Test per Multi-head Latent Attention."""

    def test_mla_output_shape(self):
        """Verifica shape output MLA."""
        from src.vision_encoder import MultiHeadLatentAttention

        mla = MultiHeadLatentAttention(
            embed_dim=256,
            num_heads=4,
            kv_rank=64
        )

        x = torch.randn(2, 100, 256)  # (B, N, D)
        output, kv_cache = mla(x)

        assert output.shape == x.shape, f"Output shape errato: {output.shape}"
        assert kv_cache is None, "kv_cache should be None when not requested"

    def test_mla_kv_cache(self):
        """Verifica che KV cache sia correttamente compresso."""
        from src.vision_encoder import MultiHeadLatentAttention

        mla = MultiHeadLatentAttention(
            embed_dim=256,
            num_heads=4,
            kv_rank=64  # Compressione da 256 a 64
        )

        x = torch.randn(2, 100, 256)
        output, kv_cache = mla(x, return_kv_cache=True)

        assert kv_cache is not None, "KV cache non ritornato"
        assert kv_cache.shape == (2, 100, 64), \
            f"KV cache shape errato: {kv_cache.shape}, expected (2, 100, 64)"

    def test_mla_attention_mask(self):
        """Verifica che attention mask funzioni."""
        from src.vision_encoder import MultiHeadLatentAttention

        mla = MultiHeadLatentAttention(
            embed_dim=256,
            num_heads=4,
            kv_rank=64
        )

        x = torch.randn(2, 10, 256)
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0  # Maschera seconda metà

        output, _ = mla(x, attention_mask=mask)

        assert output.shape == x.shape, "Output shape con mask errato"


class TestVisionPPO:
    """Test per VisionPPO network."""

    def test_forward_pass(self):
        """Verifica forward pass completo."""
        from src.vision_encoder import VisionPPO

        model = VisionPPO(
            n_actions=9,
            input_channels=4,
            embed_dim=256,
            num_heads=4,
            kv_rank=64,
            num_mla_layers=2
        )

        # Input Game Boy: (B, 4, 144, 160)
        x = torch.randn(2, 4, 144, 160)
        policy_logits, value = model(x)

        assert policy_logits.shape == (2, 9), \
            f"Policy logits shape errato: {policy_logits.shape}"
        assert value.shape == (2, 1), \
            f"Value shape errato: {value.shape}"

    def test_specialized_variants(self):
        """Verifica varianti specializzate."""
        from src.vision_encoder import (
            ExplorationPPO,
            BattlePPO,
            MenuPPO
        )

        x = torch.randn(1, 4, 144, 160)

        for ModelClass, name in [
            (ExplorationPPO, "Exploration"),
            (BattlePPO, "Battle"),
            (MenuPPO, "Menu")
        ]:
            model = ModelClass(n_actions=9)
            policy, value = model(x)

            assert policy.shape == (1, 9), f"{name}: policy shape errato"
            assert value.shape == (1, 1), f"{name}: value shape errato"

    def test_gradients_flow(self):
        """Verifica che i gradienti fluiscano correttamente."""
        from src.vision_encoder import VisionPPO

        model = VisionPPO(n_actions=9)
        x = torch.randn(2, 4, 144, 160, requires_grad=True)

        policy, value = model(x)
        loss = policy.sum() + value.sum()
        loss.backward()

        # Verifica che tutti i parametri abbiano gradienti
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradiente mancante per {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in gradiente per {name}"


class TestPPONetworkGroup:
    """Test per PPONetworkGroup."""

    def test_network_creation(self):
        """Verifica creazione corretta delle reti."""
        from src.models import PPONetworkGroup, ExplorationPPO, BattlePPO, MenuPPO

        device = torch.device('cpu')
        group = PPONetworkGroup(9, device)

        assert isinstance(group.exploration_network, ExplorationPPO)
        assert isinstance(group.battle_network, BattlePPO)
        assert isinstance(group.menu_network, MenuPPO)

    def test_network_selection(self):
        """Verifica selezione corretta della rete per stato di gioco."""
        from src.models import PPONetworkGroup

        device = torch.device('cpu')
        group = PPONetworkGroup(9, device)

        net_battle, _ = group.select_network('battle')
        net_menu, _ = group.select_network('menu')
        net_explore, _ = group.select_network('exploring')

        assert net_battle is group.battle_network
        assert net_menu is group.menu_network
        assert net_explore is group.exploration_network

    def test_choose_action(self):
        """Verifica choose_action funzioni correttamente."""
        from src.models import PPONetworkGroup

        device = torch.device('cpu')
        group = PPONetworkGroup(9, device)

        x = torch.randn(4, 144, 160)

        for game_state in ['exploring', 'battle', 'menu', 'dialogue']:
            action, log_prob, value = group.choose_action(x, game_state)

            assert 0 <= action < 9, f"{game_state}: azione fuori range"
            assert isinstance(log_prob, float), f"{game_state}: log_prob non float"
            assert isinstance(value, float), f"{game_state}: value non float"

    def test_architecture_info(self):
        """Verifica informazioni architettura."""
        from src.models import PPONetworkGroup

        device = torch.device('cpu')
        group = PPONetworkGroup(9, device)

        info = group.get_architecture_info()

        assert info['architecture'] == 'deepseek_vl2'
        assert 'networks' in info
        assert 'total_parameters' in info
        assert info['total_parameters'] > 0


class TestCountParameters:
    """Test per utility count_parameters."""

    def test_count_parameters(self):
        """Verifica conteggio parametri."""
        from src.vision_encoder import VisionPPO, count_parameters

        model = VisionPPO(n_actions=9)
        info = count_parameters(model)

        assert 'total' in info
        assert 'trainable' in info
        assert 'components' in info
        assert info['total'] == info['trainable']  # No frozen params
        assert info['total'] > 0

    def test_parameter_efficiency(self):
        """Verifica che l'architettura sia efficiente in parametri."""
        from src.models import PPONetworkGroup

        device = torch.device('cpu')
        group = PPONetworkGroup(9, device)
        info = group.get_architecture_info()

        # DeepSeek-VL2 dovrebbe avere ~3M parametri totali (vs ~20M originali)
        assert info['total_parameters'] < 5_000_000, \
            f"Troppi parametri: {info['total_parameters']:,}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
