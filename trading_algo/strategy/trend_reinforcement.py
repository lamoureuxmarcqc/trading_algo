
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class TrendLabel(str, Enum):
    DOWN = "downtrend"
    SIDEWAYS = "sideways"
    UP = "uptrend"


class ZoneLabel(str, Enum):
    BAS = "bas"
    MILIEU = "milieu"
    HAUT = "haut"


class ReinforcementSignal(str, Enum):
    STRONG_BUY = "strong_buy_zone"
    BUY = "buy_zone"
    HOLD = "hold_zone"
    AVOID_OR_TRIM = "avoid_or_trim"


@dataclass
class SecuritySnapshot:
    symbol: str
    price: float
    low_52w: float
    high_52w: float
    pe_ratio: Optional[float] = None
    prev_close: Optional[float] = None
    open_price: Optional[float] = None
    day_low: Optional[float] = None
    day_high: Optional[float] = None
    volume: Optional[int] = None
    avg_volume: Optional[int] = None


@dataclass
class TrendAnalysisResult:
    symbol: str
    trend: TrendLabel
    zone_52w: ZoneLabel
    reinforcement_signal: ReinforcementSignal
    distance_from_low_pct: float
    distance_from_high_pct: float
    palier_levels: Dict[str, float]
    notes: List[str]


class TrendReinforcementAnalyzer:
    """
    Analyseur simple pour:
    - qualifier la tendance (down/sideways/up)
    - positionner le titre dans son range 52 semaines
    - générer des signaux de renforcement par paliers
    """

    def __init__(
        self,
        low_zone_threshold: float = 0.25,
        high_zone_threshold: float = 0.75,
        strong_buy_distance_from_low_pct: float = 0.05,
        buy_distance_from_low_pct: float = 0.15,
        expensive_pe_threshold: float = 30.0,
    ):
        """
        :param low_zone_threshold: fraction du range 52w en dessous de laquelle on considère la zone "bas"
        :param high_zone_threshold: fraction du range 52w au-dessus de laquelle on considère la zone "haut"
        :param strong_buy_distance_from_low_pct: distance max au bas 52w pour signal "strong_buy_zone"
        :param buy_distance_from_low_pct: distance max au bas 52w pour signal "buy_zone"
        :param expensive_pe_threshold: seuil de C/B au-dessus duquel on devient plus prudent
        """
        self.low_zone_threshold = low_zone_threshold
        self.high_zone_threshold = high_zone_threshold
        self.strong_buy_distance_from_low_pct = strong_buy_distance_from_low_pct
        self.buy_distance_from_low_pct = buy_distance_from_low_pct
        self.expensive_pe_threshold = expensive_pe_threshold

    def analyze(self, snap: SecuritySnapshot) -> TrendAnalysisResult:
        notes: List[str] = []

        # 1. Position relative dans le range 52 semaines
        range_52w = snap.high_52w - snap.low_52w
        if range_52w <= 0:
            raise ValueError("high_52w must be greater than low_52w")

        position = (snap.price - snap.low_52w) / range_52w  # 0 = bas, 1 = haut
        distance_from_low_pct = (snap.price - snap.low_52w) / snap.low_52w
        distance_from_high_pct = (snap.high_52w - snap.price) / snap.high_52w

        if position <= self.low_zone_threshold:
            zone = ZoneLabel.BAS
            notes.append("Prix dans la zone basse du range 52 semaines.")
        elif position >= self.high_zone_threshold:
            zone = ZoneLabel.HAUT
            notes.append("Prix dans la zone haute du range 52 semaines.")
        else:
            zone = ZoneLabel.MILIEU
            notes.append("Prix dans la zone médiane du range 52 semaines.")

        # 2. Tendance simple (basée sur la position + dynamique journalière)
        trend = self._infer_trend(snap, position, notes)

        # 3. Signal de renforcement
        reinforcement_signal = self._compute_reinforcement_signal(
            snap, zone, distance_from_low_pct, notes
        )

        # 4. Paliers de renforcement (ex: 3 niveaux)
        palier_levels = self._generate_palier_levels(snap)

        return TrendAnalysisResult(
            symbol=snap.symbol,
            trend=trend,
            zone_52w=zone,
            reinforcement_signal=reinforcement_signal,
            distance_from_low_pct=distance_from_low_pct,
            distance_from_high_pct=distance_from_high_pct,
            palier_levels=palier_levels,
            notes=notes,
        )

    def _infer_trend(
        self,
        snap: SecuritySnapshot,
        position: float,
        notes: List[str],
    ) -> TrendLabel:
        """
        Tendance très simplifiée:
        - si prix proche du bas + pression vendeuse intraday -> downtrend
        - si prix proche du haut + pression acheteuse -> uptrend
        - sinon -> sideways
        """
        day_trend = None
        if snap.prev_close is not None:
            day_change = (snap.price - snap.prev_close) / snap.prev_close
            if day_change <= -0.01:
                day_trend = "down"
                notes.append(f"Baisse journalière significative: {day_change:.2%}.")
            elif day_change >= 0.01:
                day_trend = "up"
                notes.append(f"Hausse journalière significative: {day_change:.2%}.")

        # Heuristique simple
        if position < 0.3 and day_trend == "down":
            notes.append("Prix bas dans le range + pression vendeuse: tendance baissière.")
            return TrendLabel.DOWN
        if position > 0.7 and day_trend == "up":
            notes.append("Prix élevé dans le range + pression acheteuse: tendance haussière.")
            return TrendLabel.UP

        notes.append("Tendance globale considérée comme latérale/indéterminée.")
        return TrendLabel.SIDEWAYS

    def _compute_reinforcement_signal(
        self,
        snap: SecuritySnapshot,
        zone: ZoneLabel,
        distance_from_low_pct: float,
        notes: List[str],
    ) -> ReinforcementSignal:
        """
        Logique de renforcement:
        - très proche du bas 52w -> strong_buy_zone
        - bas du range -> buy_zone
        - milieu -> hold_zone
        - haut du range -> avoid_or_trim
        Ajusté par la cherté (C/B).
        """
        signal: ReinforcementSignal

        if distance_from_low_pct <= self.strong_buy_distance_from_low_pct:
            signal = ReinforcementSignal.STRONG_BUY
            notes.append("Très proche du bas 52 semaines: zone de renforcement fort.")
        elif distance_from_low_pct <= self.buy_distance_from_low_pct:
            signal = ReinforcementSignal.BUY
            notes.append("Proche du bas 52 semaines: zone de renforcement.")
        elif zone == ZoneLabel.MILIEU:
            signal = ReinforcementSignal.HOLD
            notes.append("Zone médiane: conserver, renforcement sélectif seulement.")
        else:  # Zone haute
            signal = ReinforcementSignal.AVOID_OR_TRIM
            notes.append("Zone haute: éviter de renforcer, envisager de réduire si surpondéré.")

        # Ajustement par le C/B si disponible
        if snap.pe_ratio is not None and snap.pe_ratio > self.expensive_pe_threshold:
            notes.append(
                f"C/B élevé ({snap.pe_ratio:.1f}) > {self.expensive_pe_threshold}: prudence accrue."
            )
            if signal in (ReinforcementSignal.STRONG_BUY, ReinforcementSignal.BUY):
                # On dégrade d'un cran
                if signal == ReinforcementSignal.STRONG_BUY:
                    signal = ReinforcementSignal.BUY
                    notes.append("Signal réduit de strong_buy à buy à cause du C/B.")
                elif signal == ReinforcementSignal.BUY:
                    signal = ReinforcementSignal.HOLD
                    notes.append("Signal réduit de buy à hold à cause du C/B.")

        return signal

    def _generate_palier_levels(self, snap: SecuritySnapshot) -> Dict[str, float]:
        """
        Génère des niveaux de prix pour tes paliers de renforcement.
        Exemple:
        - palier_1: très proche du bas 52w
        - palier_2: bas + 1/3 du range
        - palier_3: bas + 2/3 du range
        """
        low = snap.low_52w
        high = snap.high_52w
        r = high - low

        return {
            "palier_1": round(low * 1.02, 2),          # +2% au-dessus du bas
            "palier_2": round(low + r * (1/3), 2),
            "palier_3": round(low + r * (2/3), 2),
        }
if "__main__":
    # Exemple d'utilisation
    snap = SecuritySnapshot(
        symbol="EXAMPLE",
        price=95.0,
        low_52w=80.0,
        high_52w=120.0,
        pe_ratio=25.0,
        prev_close=100.0,
    )
    analyzer = TrendReinforcementAnalyzer()
    result = analyzer.analyze(snap)
    print(result)