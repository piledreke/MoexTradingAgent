import pytest
from aioresponses import aioresponses

from utils.moex_client import MoexClient


@pytest.mark.asyncio
async def test_to_dataframe_classic_format():
    payload = {
        "securities": {
            "columns": ["SECID", "LAST", "VOLTODAY"],
            "data": [["SBER", "312.5", "1000"], ["GAZP", "150.0", "2000"]],
        }
    }
    df = MoexClient.to_dataframe(payload, "securities")
    assert len(df) == 2
    assert df["SECID"].tolist() == ["SBER", "GAZP"]
    assert df["LAST"].dtype != object  # coerced to numeric


@pytest.mark.asyncio
async def test_get_share_uses_public():
    client = MoexClient(algopack_token=None, cache_ttl=0)
    url = ("https://iss.moex.com/iss/engines/stock/markets/shares/boards/tqbr"
           "/securities/SBER.json")
    payload = {
        "securities": {"columns": ["SECID"], "data": [["SBER"]]},
        "marketdata": {"columns": ["LAST"], "data": [["312.5"]]},
    }
    with aioresponses() as m:
        m.get(url, payload=payload)
        result = await client.get_share("SBER")
    assert result["securities"]["SECID"] == "SBER"
    await client.close()


@pytest.mark.asyncio
async def test_request_retries_on_500():
    client = MoexClient(algopack_token=None, cache_ttl=0)
    url = "https://iss.moex.com/iss/securities/SBER.json"
    with aioresponses() as m:
        m.get(url, status=500)
        m.get(url, status=500)
        m.get(url, payload={"description": {"columns": ["name", "value"],
                                            "data": [["SHORTNAME", "Сбербанк"]]}})
        info = await client.get_security_info("SBER")
    assert info.get("SHORTNAME") == "Сбербанк"
    await client.close()