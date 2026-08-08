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


class _TinyPositiveInterTimeModel(_ZeroInterTimeModel):
    def __init__(self):
        self.calls = 0

    def sample_next_inter_time(
        self,
        inter_time_dist,
        t_last_event=None,
        lower_bound=None,
        max_inter_time=None,
    ):
        del inter_time_dist, t_last_event, lower_bound
        self.calls += 1
        if self.calls == 1:
            return torch.tensor([[1.0]], device=self.device)
        if self.calls == 2:
            return torch.tensor([[1e-8]], device=self.device)
        return torch.as_tensor(max_inter_time, device=self.device).reshape(1, 1)


def test_sampling_terminates_when_inter_time_does_not_advance():
    model = _ZeroInterTimeModel()

    sampled = model.sample(
        batch_size=2,
        duration=1.0,
        return_sequences=True,
    )

    assert len(sampled) == 2
    assert all(len(sequence) == 0 for sequence in sampled)


def test_sampling_keeps_tiny_positive_inter_time_as_event():
    model = _TinyPositiveInterTimeModel()

    sampled = model.sample(
        batch_size=1,
        duration=2.0,
        return_sequences=True,
    )

    sequence = sampled[0]
    assert len(sequence) == 2
    assert torch.all(sequence.inter_times[:-1] > 0.0)
    assert sequence.arrival_times[1] > sequence.arrival_times[0]
