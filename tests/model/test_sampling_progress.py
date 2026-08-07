import torch

from src.models.tpp.recurrent.sampling import RecurrentTPPSamplingMixin


class _ZeroInterTimeModel(RecurrentTPPSamplingMixin):
    context_size = 1
    device = torch.device("cpu")
    input_magnitude = False
    num_extra_features = None
    num_rnn_layers = 1

    def _sampling_state_step(self, rnn_input, current_hidden):
        del rnn_input
        next_state = current_hidden.new_zeros(
            current_hidden.shape[1],
            1,
            self.context_size,
        )
        return next_state, current_hidden

    def encode_time(self, inter_times):
        return inter_times

    def get_inter_time_dist(self, time_context):
        del time_context
        return None

    def get_magnitude_dist(self, *args, **kwargs):
        del args, kwargs
        return None

    def sample_next_inter_time(
        self,
        inter_time_dist,
        t_last_event=None,
        lower_bound=None,
        max_inter_time=None,
    ):
        del inter_time_dist, t_last_event, lower_bound, max_inter_time
        return torch.zeros(2, 1, device=self.device)


def test_sampling_terminates_when_inter_time_does_not_advance():
    model = _ZeroInterTimeModel()

    sampled = model.sample(
        batch_size=2,
        duration=1.0,
        return_sequences=True,
    )

    assert len(sampled) == 2
    assert all(len(sequence) == 0 for sequence in sampled)
