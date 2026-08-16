from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    component_id: Mapped[str] = mapped_column(index=True)
    timestamp: Mapped[str]
    predicted_condition: Mapped[str]
    probability: Mapped[float]
    feature_data: Mapped[str]
    shap_explanation: Mapped[str]
    edge_device_id: Mapped[str]
    created_at: Mapped[str]
