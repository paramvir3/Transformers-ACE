from train import split_trajectory_frames


def test_blocked_split_keeps_whole_blocks_and_boundary_gap():
    frames = list(range(100))
    train, validation, excluded = split_trajectory_frames(
        frames,
        val_fraction=0.2,
        seed=42,
        mode="blocked",
        block_size=10,
        gap=2,
    )

    validation_set = set(validation)
    train_set = set(train)
    excluded_set = set(excluded)
    assert validation_set
    assert train_set
    assert not (validation_set & train_set)
    assert not (excluded_set & train_set)
    assert not (excluded_set & validation_set)

    for start in range(0, 100, 10):
        block = set(range(start, start + 10))
        assert block <= validation_set or not (block & validation_set)

    for index in validation:
        nearby = set(range(max(0, index - 2), min(100, index + 3)))
        assert not (nearby & train_set)


def test_split_is_deterministic_for_a_fixed_seed():
    frames = list(range(60))
    first = split_trajectory_frames(frames, seed=9, block_size=10)
    second = split_trajectory_frames(frames, seed=9, block_size=10)
    assert first == second


def test_blocked_split_adapts_to_a_dataset_smaller_than_one_configured_block():
    train, validation, excluded = split_trajectory_frames(
        list(range(8)),
        val_fraction=0.25,
        seed=2,
        mode="blocked",
        block_size=25,
    )
    assert train
    assert validation
    assert not excluded
