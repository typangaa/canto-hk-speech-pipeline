from pipeline.nodes.label_store import bucket_pitch, bucket_rate


def test_bucket_rate_slow_normal_fast():
    assert bucket_rate(3.0, p25=4.0, p75=5.0) == "slow"
    assert bucket_rate(4.5, p25=4.0, p75=5.0) == "normal"
    assert bucket_rate(6.0, p25=4.0, p75=5.0) == "fast"


def test_bucket_rate_boundary_values_are_normal():
    assert bucket_rate(4.0, p25=4.0, p75=5.0) == "normal"
    assert bucket_rate(5.0, p25=4.0, p75=5.0) == "normal"


def test_bucket_pitch_low_normal_high():
    z, bucket = bucket_pitch(100.0, mu=150.0, sigma=20.0)
    assert bucket == "low"
    assert z < -0.5

    z, bucket = bucket_pitch(150.0, mu=150.0, sigma=20.0)
    assert bucket == "normal"
    assert z == 0.0

    z, bucket = bucket_pitch(200.0, mu=150.0, sigma=20.0)
    assert bucket == "high"
    assert z > 0.5


def test_bucket_pitch_sigma_zero_does_not_crash():
    z, bucket = bucket_pitch(160.0, mu=150.0, sigma=0.0)
    assert isinstance(z, float)
    assert bucket in {"low", "normal", "high"}
