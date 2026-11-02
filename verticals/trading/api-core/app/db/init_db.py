from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password
from app.db.base import Base
from app.db.models import Account, AuditLog, CashBalance, Fill, Order, Position, Role, Symbol, User, UserRole
from app.db.session import SessionLocal, engine


def initialize_database(seed_demo_data: bool = True) -> None:
    Base.metadata.create_all(bind=engine)
    if not seed_demo_data:
        return

    with SessionLocal() as session:
        seed_database(session)


def seed_database(session: Session) -> None:
    roles = _ensure_roles(session)
    _ensure_demo_users(session, roles)

    if session.scalar(select(Account.id).limit(1)):
        _ensure_audit_seed(session)
        session.commit()
        return

    user = session.scalar(select(User).where(User.email == settings.default_user_email))
    if not user:
        raise ValueError("Default CIO user was not created")

    account = Account(
        broker_name="Interactive Brokers",
        account_number="FO-001",
        account_type="family_office",
        base_currency=settings.default_base_currency,
        status="active",
    )
    session.add(account)
    session.flush()

    session.add(CashBalance(account_id=account.id, currency=settings.default_base_currency, balance=Decimal("412000.00")))

    symbols = {
        "AAPL": Symbol(ticker="AAPL", asset_class="equity", exchange="NASDAQ", currency="USD"),
        "MSFT": Symbol(ticker="MSFT", asset_class="equity", exchange="NASDAQ", currency="USD"),
        "NVDA": Symbol(ticker="NVDA", asset_class="equity", exchange="NASDAQ", currency="USD"),
    }
    session.add_all(list(symbols.values()))
    session.flush()

    positions = [
        Position(
            account_id=account.id,
            symbol_id=symbols["AAPL"].id,
            quantity=Decimal("1200"),
            average_cost=Decimal("178.40"),
            market_price=Decimal("191.80"),
            market_value=Decimal("230160.00"),
            unrealized_pnl=Decimal("16080.00"),
        ),
        Position(
            account_id=account.id,
            symbol_id=symbols["MSFT"].id,
            quantity=Decimal("760"),
            average_cost=Decimal("404.20"),
            market_price=Decimal("417.50"),
            market_value=Decimal("317300.00"),
            unrealized_pnl=Decimal("10108.00"),
        ),
        Position(
            account_id=account.id,
            symbol_id=symbols["NVDA"].id,
            quantity=Decimal("500"),
            average_cost=Decimal("834.50"),
            market_price=Decimal("912.00"),
            market_value=Decimal("456000.00"),
            unrealized_pnl=Decimal("38750.00"),
        ),
    ]
    session.add_all(positions)
    session.flush()

    order = Order(
        account_id=account.id,
        symbol_id=symbols["MSFT"].id,
        side="BUY",
        order_type="limit",
        quantity=Decimal("100"),
        limit_price=Decimal("415.20"),
        status="filled",
        broker_order_id="seed-msft-1",
        strategy_tag="seed",
    )
    session.add(order)
    session.flush()
    session.add(
        Fill(
            order_id=order.id,
            venue="IEX",
            quantity=Decimal("100"),
            price=Decimal("415.20"),
            fees=Decimal("1.25"),
        )
    )
    _ensure_audit_seed(session)
    session.commit()


def _ensure_roles(session: Session) -> dict[str, Role]:
    role_specs = {
        "admin": "Platform administrator",
        "trader": "Execution and OMS access",
        "analyst": "Research and signal analysis",
        "read-only": "Read-only dashboard access",
        "risk_officer": "Risk oversight and limit monitoring",
    }
    roles: dict[str, Role] = {}
    for name, description in role_specs.items():
        role = session.scalar(select(Role).where(Role.name == name))
        if not role:
            role = Role(name=name, description=description)
            session.add(role)
            session.flush()
        roles[name] = role
    return roles


def _ensure_demo_users(session: Session, roles: dict[str, Role]) -> None:
    demo_users = [
        {
            "email": settings.default_user_email,
            "full_name": "Chief Investment Officer",
            "role": "admin",
            "mfa_secret": "demo-mfa-secret",
        },
        {
            "email": "trader@hedgefund.local",
            "full_name": "Lead Trader",
            "role": "trader",
            "mfa_secret": None,
        },
        {
            "email": "research@hedgefund.local",
            "full_name": "Research Analyst",
            "role": "analyst",
            "mfa_secret": None,
        },
        {
            "email": "risk@hedgefund.local",
            "full_name": "Risk Officer",
            "role": "risk_officer",
            "mfa_secret": None,
        },
        {
            "email": "viewer@hedgefund.local",
            "full_name": "Investor Relations",
            "role": "read-only",
            "mfa_secret": None,
        },
    ]

    for spec in demo_users:
        user = session.scalar(select(User).where(User.email == spec["email"]))
        if not user:
            user = User(
                email=spec["email"],
                full_name=spec["full_name"],
                password_hash=hash_password("demo"),
                mfa_secret=spec["mfa_secret"],
                is_active=True,
            )
            session.add(user)
            session.flush()

        role = roles[spec["role"]]
        membership = session.scalar(
            select(UserRole).where(UserRole.user_id == user.id, UserRole.role_id == role.id)
        )
        if not membership:
            session.add(UserRole(user_id=user.id, role_id=role.id))


def _ensure_audit_seed(session: Session) -> None:
    if session.scalar(select(AuditLog.id).limit(1)):
        return
    session.add_all(
        [
            AuditLog(
                event_type="platform.bootstrap",
                entity_type="system",
                entity_id="api-core",
                actor_email="system",
                details="Initial platform bootstrap completed",
            ),
            AuditLog(
                event_type="portfolio.seed",
                entity_type="portfolio",
                entity_id="family-office-master",
                actor_email="system",
                details="Demo family office portfolio seeded",
            ),
        ]
    )
