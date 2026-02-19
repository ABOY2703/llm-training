from llm_trainer.train.loop import compute_rates, make_batches


def test_compute_rates():
    it_s, tok_s = compute_rates(0.5, 200)
    assert round(it_s, 2) == 2.0
    assert round(tok_s, 2) == 400.0


def test_make_batches_shape():
    ds = make_batches(list(range(40)), seq_len=9)
    x, y = ds[0]
    assert x.shape[0] == 9
    assert y.shape[0] == 9
