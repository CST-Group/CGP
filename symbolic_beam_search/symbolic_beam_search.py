from concurrent.futures import ThreadPoolExecutor
import math
import torch
import torch.nn.functional as F


class SymbolicBeamSearch:

    def __init__(self, sdr_idea_deserializer):
        self.sdr_idea_deserializer = sdr_idea_deserializer

    def perform_beam_search(
        self,
        model,
        src,
        start_symbol,
        end_symbol,
        max_len,
        beam_size,
        temperature,
        device,
    ):
        src = src.to(device)
        ys = torch.LongTensor([[start_symbol]]).type_as(src.data)
        states = self._initialize_states(False)
        steps = self._initialize_steps(states[0])
        beam = [(ys, 0.0, 7.0, False, states, steps)]
        finished_beams = []

        self.sdr_idea_deserializer.convert_dictionary_key_to_string()

        for _ in range(max_len):
            beam_candidates = self._expand_beam(
                beam, src, model, temperature, beam_size, end_symbol, device
            )
            beam = self._select_top_beams(beam_candidates, beam_size)

            if not beam:
                break

        if not finished_beams:
            finished_beams = beam

        return finished_beams

    def _expand_beam(
        self, beam, src, model, temperature, beam_size, end_symbol, device
    ):
        beam_candidates = []
        with ThreadPoolExecutor(max_workers=beam_size) as executor:
            for answer, score, finished, size, states, steps in executor.map(
                lambda answer: self._process_beam_item(
                    answer, src, model, temperature, beam_size, end_symbol, device
                ),
                beam,
            ):
                if finished:
                    if size > 100:
                        beam_candidates.append(
                            (answer, score, finished, size, states, steps)
                        )
                else:
                    beam_candidates.extend(answer)
        return beam_candidates

    def _select_top_beams(self, beam_candidates, beam_size):
        beam_candidates.sort(key=lambda x: x[1])
        return beam_candidates[:beam_size]

    def _process_beam_item(
        self, item, src, model, temperature, beam_size, end_symbol, device="cpu"
    ):
        answer, score, _, _, states, steps = item
        if answer[-1, -1].eq(end_symbol).item():
            return answer, score, True, answer.shape[1], states, steps

        current_state, current_step, states, steps, allowed_tokens = (
            self._get_allowed_tokens(states, steps)
        )

        with torch.no_grad():
            logits = model(src, answer)[-1, :]
        logits.div_(temperature)
        probs = F.softmax(logits, dim=-1)

        if allowed_tokens is not None:
            probs = self._filter_allowed_tokens(probs, allowed_tokens)

        top_limit = min(beam_size, len(allowed_tokens))
        top_probs, top_ix = probs.topk(top_limit)

        candidates = self._generate_candidates(
            answer, score, top_probs, top_ix, states, steps, device
        )

        return candidates, score, False, answer.shape[1], states, steps

    def _filter_allowed_tokens(self, probs, allowed_tokens):
        for index in range(len(probs[0, :])):
            if index not in allowed_tokens:
                probs[:, index] = 0.0
        return probs

    def _generate_candidates(
        self, answer, score, top_probs, top_ix, states, steps, device
    ):
        candidates = []
        for i in range(len(top_probs[-1])):
            prob = top_probs[-1, i].item()
            ix = top_ix[-1, i].item()
            next_answer = torch.cat(
                [answer, torch.tensor([[ix]], device=device)], dim=1
            )
            next_score = score - math.log(prob)

            new_steps = steps.copy()
            new_states = states.copy()

            self._validate_and_extend_steps(
                new_states, new_steps, states[0], next_answer
            )

            candidates.append(
                (
                    next_answer,
                    next_score,
                    False,
                    next_answer.shape[1],
                    new_states,
                    new_steps,
                )
            )

        return candidates

    def _initialize_states(self, is_parent=False):
        sequence = ["ID", "NAME", "TYPE", "METADATA", "LENGTH"]
        if is_parent:
            sequence.insert(0, "PARENT_ID")

        return sequence

    def _initialize_steps(self, state):

        steps = {
            "PARENT_ID": ["END", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
            "ID": ["NUMBER", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
            "NAME": ["STRING"],
            "TYPE": ["TYPE"],
            "METADATA": ["METADATA"],
            "LENGTH": ["NUMBER", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
            "NUM_VALUE": ["NUMBER", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
            "STRING_VALUE": ["STRING"],
            "END": ["END"],
        }

        return steps[state]

    def _get_allowed_tokens(self, states, steps):
        current_state = None
        current_step = None

        if len(steps) == 0:
            states.pop(0)

            if len(states) == 0:
                states = self._initialize_states(True)

            current_state = states[0]
            steps = self._initialize_steps(current_state)

        if current_state is None:
            current_state = states[0]

        current_step = steps.pop(0)
        allowed_tokens = self._get_step_index()[current_step]

        return current_state, current_step, states, steps, allowed_tokens

    def _get_step_index(self):
        states = {
            "NUMBER": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "SIGNAL": [4, 5],
            "STRING": [16, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32],
            "TYPE": [17],
            "METADATA": [17, 18, 21, 26],
            "SPECIAL": [2],
            "END": [2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        }

        return states

    def _validate_and_extend_steps(self, states, steps, current_state, next_answer):
        if len(steps) == 0:
            if current_state == "LENGTH":
                length_value = self.sdr_idea_deserializer.get_local_numeric_value(
                    next_answer[0, -6:].tolist()
                )

                if length_value <= 0:
                    length_value = 1

                metadata_type = self.sdr_idea_deserializer.get_metadata_type(
                    int(
                        self.sdr_idea_deserializer.get_local_string_value(
                            next_answer[:, -7].item()
                        )
                    )
                )
                if metadata_type == "STRING_ARRAY" or metadata_type == "STRING_VALUE":
                    steps.extend(
                        int(length_value) * self._initialize_steps("STRING_VALUE")
                    )
                elif metadata_type == "NUM_ARRAY" or metadata_type == "NUM_VALUE":
                    steps.extend(
                        int(length_value) * self._initialize_steps("NUM_VALUE")
                    )

                states.pop(0)
                states.append("VALUE")
