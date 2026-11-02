from __future__ import annotations

from decimal import Decimal
import sys
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.db.base import Base  # noqa: E402
from app.db.models import Account, Position, Symbol  # noqa: E402
from app.schemas.admin import AdminSymbolMarketDataUpdate  # noqa: E402
from app.services.platform_service import PlatformService  # noqa: E402
from app.services.quant_service import quant_insights_service  # noqa: E402
from scripts.import_portfolio_csv import load_rows  # noqa: E402


def _build_service() -> tuple[Session, PlatformService]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(bind=engine)
    account = Account(
        broker_name="Imported CSV",
        account_number="ACCT-001",
        account_type="brokerage",
        base_currency="CAD",
        status="active",
    )
    session.add(account)
    session.flush()

    enabled = Symbol(
        ticker="BN.TO",
        asset_class="equity",
        exchange="TSX",
        market_data_ticker="BN.TO",
        market_data_enabled=True,
        currency="CAD",
    )
    disabled = Symbol(
        ticker="RBF2011.TO",
        asset_class="mutual_fund",
        exchange="TSX",
        market_data_ticker=None,
        market_data_enabled=False,
        currency="CAD",
    )
    session.add_all([enabled, disabled])
    session.flush()

    session.add_all(
        [
            Position(
                account_id=account.id,
                symbol_id=enabled.id,
                quantity=Decimal("10"),
                average_cost=Decimal("60"),
                market_price=Decimal("62"),
                market_value=Decimal("620"),
                unrealized_pnl=Decimal("20"),
            ),
            Position(
                account_id=account.id,
                symbol_id=disabled.id,
                quantity=Decimal("100"),
                average_cost=Decimal("10"),
                market_price=Decimal("10"),
                market_value=Decimal("1000"),
                unrealized_pnl=Decimal("0"),
            ),
        ]
    )
    session.commit()
    return session, PlatformService(session)


def test_load_rows_classifies_non_quoted_sections_and_normalizes_tickers(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text(
        "\n".join(
            [
                "sep=;",
                "Compte;Symbole;Nom;Qte;Cout Moyen $;Cout Total $;Heure differe;Prix actuel $;Variation du jour $;Variation du jour %;Qte x valeur du jour $;Valeur au marche $;Valeur d?emprunt $;Profits non realises $;Profits non realises %",
                "ACTIONS CAD",
                "ACC1;BN-C;BROOKFIELD CORP CL-A LVS;12;58,408;700,90;2026-05-20 16:00:02.0;62,29;1,4;2,299;0;747,48;N/D;46,58;6,646",
                "TITRES A REVENU FIXE CAD",
                "ACC1;K25VH9-C;CPG BLC 5.00%CA 22NV27;100;100;10000;2026-05-19 00:00:00.0;100;0;0;0;10000;N/D;0;0",
                "FONDS COMMUNS CAD",
                "ACC1;RBF2011-C;RBC CP EP PLC-F;100;10;1000;2026-05-19 00:00:00.0;10;0;0;0;1000;N/D;0;0",
                "ACTIONS USD",
                "ACC1;MSFT-U;MICROSOFT CORP;5;419,877;2099,39;2026-05-20 16:00:55.0;421;3,58;0,858;0;2105;N/D;5,61;0,267",
            ]
        ),
        encoding="cp1252",
    )

    rows = load_rows(csv_path)

    assert [row.ticker for row in rows] == ["BN.TO", "K25VH9.TO", "RBF2011.TO", "MSFT"]
    assert rows[0].asset_class == "equity"
    assert rows[0].market_data_enabled is True
    assert rows[1].asset_class == "fixed_income"
    assert rows[1].market_data_enabled is False
    assert rows[2].asset_class == "mutual_fund"
    assert rows[2].market_data_enabled is False
    assert rows[3].asset_class == "equity"
    assert rows[3].market_data_enabled is True


def test_refresh_positions_skips_market_data_for_disabled_symbols(monkeypatch) -> None:
    session, service = _build_service()
    captured_requests: dict[str, str | None] = {}

    def fake_snapshot(requests: dict[str, str | None]) -> dict[str, dict[str, float | None]]:
        captured_requests.update(requests)
        return {
            "BN.TO": {"latest_price": 63.5, "previous_close": 62.0},
            "RBF2011.TO": {"latest_price": None, "previous_close": None},
        }

    monkeypatch.setattr(quant_insights_service, "get_live_market_snapshot_for_requests", fake_snapshot)

    service.refresh_portfolio_market_data(force=True)
    session.expire_all()
    positions = {position.symbol.ticker: position for position in session.scalars(select(Position)).all()}

    assert captured_requests == {"BN.TO": "BN.TO", "RBF2011.TO": None}
    assert float(positions["BN.TO"].market_price) == 63.5
    assert float(positions["BN.TO"].market_value) == 635.0
    assert float(positions["RBF2011.TO"].market_price) == 10.0
    assert float(positions["RBF2011.TO"].market_value) == 1000.0


def test_admin_symbol_views_and_updates() -> None:
    _session, service = _build_service()

    unresolved = service.list_admin_symbols(unresolved_only=True)
    assert [item.ticker for item in unresolved] == ["RBF2011.TO"]

    all_symbols = service.list_admin_symbols()
    bn_entry = next(item for item in all_symbols if item.ticker == "BN.TO")
    rbf_entry = next(item for item in all_symbols if item.ticker == "RBF2011.TO")
    assert bn_entry.market_data_enabled is True
    assert bn_entry.market_data_ticker == "BN.TO"
    assert rbf_entry.market_data_enabled is False
    assert rbf_entry.position_count == 1

    updated = service.update_symbol_market_data(
        rbf_entry.id,
        AdminSymbolMarketDataUpdate(
            market_data_enabled=True,
            market_data_ticker="RBF2011-FUND",
        ),
    )
    assert updated.market_data_enabled is True
    assert updated.market_data_ticker == "RBF2011-FUND"
