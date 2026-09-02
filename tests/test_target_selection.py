from teknofest_iha.core.target_selection import choose_visible_unreleased_target


def test_red_can_be_selected_before_primary_when_it_is_the_visible_target():
    red = {
        "target_type": "red_square",
        "target_state": "LOCKED",
        "fusion_confidence": 0.82,
    }

    selected = choose_visible_unreleased_target(
        selected=red,
        targets=[red],
        allowed_targets=("blue_square", "red_square"),
        can_release=lambda target: True,
    )

    assert selected == "red_square"


def test_released_target_is_skipped_for_remaining_target():
    blue = {
        "target_type": "blue_square",
        "target_state": "DROP_READY",
        "fusion_confidence": 0.95,
    }
    red = {
        "target_type": "red_square",
        "target_state": "CANDIDATE",
        "fusion_confidence": 0.75,
    }

    selected = choose_visible_unreleased_target(
        selected=blue,
        targets=[blue, red],
        allowed_targets=("blue_square", "red_square"),
        can_release=lambda target: target != "blue_square",
    )

    assert selected == "red_square"


def test_more_mature_target_wins_over_primary_bias_when_both_are_available():
    blue = {
        "target_type": "blue_square",
        "target_state": "CANDIDATE",
        "fusion_confidence": 0.7,
    }
    red = {
        "target_type": "red_square",
        "target_state": "DROP_READY",
        "fusion_confidence": 0.95,
    }

    selected = choose_visible_unreleased_target(
        selected=blue,
        targets=[blue, red],
        allowed_targets=("blue_square", "red_square"),
        can_release=lambda target: True,
    )

    assert selected == "red_square"
