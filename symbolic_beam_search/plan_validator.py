from pyctm.representation.sdr_idea_array import SDRIdeaArray


class PlanValidator:
    def __init__(self, sdr_idea_deserializer):
        self.sdr_idea_deserializer = sdr_idea_deserializer

    def filter_valid_plans(
        self,
        occupiedNodes,
        initial_node,
        closest_nodes_from_goal_tag,
        action,
        finished_beams,
    ):
        valid_sequences = []
        for ys, score, size, _, _, _ in finished_beams:
            try:
                idea_array = self._convert_to_idea_array(ys)

                if self._is_valid_sequence(idea_array) and self._is_valid_plan(
                    action,
                    initial_node,
                    idea_array,
                    occupiedNodes,
                    closest_nodes_from_goal_tag,
                ):
                    valid_sequences.append((ys, score, size))
            except Exception:
                continue

        return valid_sequences

    def _convert_to_idea_array(self, tensor):
        sdr_idea = tensor.squeeze(0).detach().cpu().numpy().tolist()

        plan_idea_array = SDRIdeaArray(10, 7, 0)
        plan_idea_array.sdr = sdr_idea

        action_step_idea = self.sdr_idea_deserializer.deserialize(plan_idea_array)

        full_goal = [action_step_idea]
        for i in range(len(action_step_idea.child_ideas)):
            full_goal.append(action_step_idea.child_ideas[i])

        return full_goal

    def _is_valid_sequence(self, idea_array):
        for idea in idea_array:
            if not self._is_valid_idea(idea):
                return False
        return True

    def _is_valid_idea(self, step_idea):
        if step_idea.name == "pick" or step_idea.name == "place":
            if (
                isinstance(step_idea.value, list)
                and len(step_idea.value) == 2
                and 0 <= step_idea.value[0] <= 187
                and 1 <= step_idea.value[1] <= 4
            ):
                return True
            else:
                return False

        elif step_idea.name == "moveTo":
            if isinstance(step_idea.value, float) and 0 <= step_idea.value <= 187:
                return True
            else:
                return False

        elif step_idea.name == "moveToNode":
            if isinstance(step_idea.value, float) and 1 <= step_idea.value <= 16:
                return True
            else:
                return False

        return True

    def _is_valid_plan(
        self,
        action,
        initial_node,
        plan_steps,
        occupiedNodes,
        closest_nodes_from_goal_tag,
    ):
        if action == "PICK":
            current_step = plan_steps[0]
            if current_step.name == "moveToNode":
                if float(current_step.value) != float(initial_node):
                    current_node = str(current_step.value)
                    initial_node = str(initial_node)
                    if current_node not in self.get_graph_connection()[initial_node]:
                        return False

        for i in range(len(plan_steps) - 1):
            current_step = plan_steps[i]
            next_step = plan_steps[i + 1]

            if current_step.name == "moveToNode" and next_step.name == "moveToNode":
                try:
                    current_node = str(current_step.value)
                    next_node = str(next_step.value)
                    if next_node not in self.get_graph_connection()[current_node]:
                        return False

                    if current_step.value in occupiedNodes:
                        return False
                except Exception:
                    return False

        return True

    def get_graph_connection(self):
        graph_connection = {
            "1.0": ["2.0", "16.0"],
            "2.0": ["1.0", "3.0", "15.0"],
            "3.0": ["2.0", "4.0", "14.0"],
            "4.0": ["3.0", "5.0"],
            "5.0": ["4.0", "6.0", "14.0"],
            "6.0": ["5.0", "7.0", "13.0"],
            "7.0": ["6.0", "8.0"],
            "8.0": ["7.0", "9.0", "12.0"],
            "9.0": ["8.0", "10.0"],
            "10.0": ["9.0", "11.0", "16.0"],
            "11.0": ["10.0", "12.0", "15.0"],
            "12.0": ["11.0", "13.0", "8.0"],
            "13.0": ["6.0", "12.0", "14.0"],
            "14.0": ["3.0", "5.0", "13.0", "15.0"],
            "15.0": ["2.0", "11.0", "14.0", "16.0"],
            "16.0": ["1.0", "10.0", "15.0"],
        }

        return graph_connection
