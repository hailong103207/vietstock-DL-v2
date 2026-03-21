# User bots

Thư mục `user/bots/` dành cho bot do người dùng tự định nghĩa.

## Cách bắt đầu

1. Mở `user/builtin/bots/technical/` để xem bot/technique mẫu.
2. Mở `user/bots/examples/examples.py` để xem bot mẫu chạy được ngay với backtest engine.
3. Đổi tên class/technique và chỉnh logic theo nhu cầu.
4. Giữ import từ các module lõi ổn định (signal, context, backtest engine).

## Khuyến nghị tổ chức

- `user/bots/strategies/`: strategy theo từng phong cách giao dịch.
- `user/bots/confirmations/`: rule xác nhận tín hiệu.
- `user/bots/indicators/`: chỉ báo custom.
- `user/bots/pipelines/`: tổ hợp nhiều technique thành bot hoàn chỉnh.

## Lưu ý quan trọng

- Tránh đổi contract của `Signal`, `StepContext`, `Action`.
- Không hard-code path tuyệt đối; dùng path tương đối từ project root.
- Dự án chỉ dùng **một registry duy nhất** tại `src/vnstock_forecast/forecast/technical/registry.py`.
- Mỗi bot nên có:
    - mô tả rõ điều kiện vào/ra,
    - ngưỡng confidence,
    - rủi ro mặc định (SL/TP, allocation).
- Nên log lý do tạo signal để truy vết khi backtest/live scan.

## Đăng ký bot/technique

Khi tạo technique mới trong `user/bots/`, đăng ký trực tiếp vào core registry:

1. Import decorator:
    - `from vnstock_forecast.builtin.signal_based.technical.registry import register`
2. Gắn decorator lên class kế thừa `BaseTechnique`:
    - `@register("ten_technique_duy_nhat")`
3. Module trong `user/bots/` hiện được runtime auto-load, nên decorator sẽ được thực thi tự động khi hệ thống truy vấn registry.

Lưu ý:

- Tên trong `@register(...)` phải duy nhất toàn hệ thống.
- Không tạo registry riêng trong `user/*` để tránh chia tách trạng thái technique.

## Hướng dẫn viết bot (chi tiết)

Phần này bám theo `notebooks/guide/backtest_tutorial.ipynb`.

### 1) Chọn kiểu bot

- **Backtest bot thuần** (dùng trực tiếp với `BacktestEngine`): kế thừa `BotBase`, implement `on_step(ctx) -> list[Action]`.
- **Technique cho SignalBasedBacktestBot**: kế thừa `BaseTechnique`, emit `Signal`, đăng ký qua `@register(...)`.

Nếu bạn mới bắt đầu, hãy viết bot thuần trước (dễ debug hơn).

### 2) Contract bắt buộc

- `on_step(ctx)` được gọi mỗi bar.
- Không dùng future data; chỉ đọc qua `ctx.history(...)`, `ctx.latest(...)`, `ctx.price(...)`.
- Trả về danh sách `Action` hợp lệ (`BUY`/`SELL`).
- Không thao tác trực tiếp vào `Portfolio`; engine sẽ thực thi action.

### 3) Các API dùng thường xuyên từ StepContext

- `ctx.timestamp`: thời điểm bar hiện tại.
- `ctx.cash`: tiền mặt khả dụng.
- `ctx.symbols`: danh sách mã đang chạy.
- `ctx.price(symbol)`: giá Close hiện tại.
- `ctx.history(symbol, lookback=N)`: N bars gần nhất.
- `ctx.has_position(symbol)`: đang giữ mã đó hay không.
- `ctx.positions_for(symbol)`: danh sách vị thế đang mở của mã.

### 4) Mẫu bot tối thiểu

```python
from vnstock_forecast.engine.backtest.bot_base import BotBase, Action, ActionType
from vnstock_forecast.engine.backtest.context import StepContext


class MyFirstBot(BotBase):
    name = "MyFirstBot"

    def __init__(self, allocation: float = 0.3) -> None:
        self.allocation = allocation

    def on_step(self, ctx: StepContext) -> list[Action]:
        actions: list[Action] = []

        for symbol in ctx.symbols:
            df = ctx.history(symbol, lookback=21)
            if len(df) < 21:
                continue

            close = df["Close"]
            sma20 = close.rolling(20).mean()
            if sma20.isna().iloc[-1] or sma20.isna().iloc[-2]:
                continue

            price = ctx.price(symbol)
            prev_close = close.iloc[-2]
            prev_sma = sma20.iloc[-2]
            curr_sma = sma20.iloc[-1]

            if prev_close <= prev_sma and price > curr_sma and not ctx.has_position(symbol):
                qty = int(ctx.cash * self.allocation // price)
                if qty > 0:
                    actions.append(
                        Action(
                            type=ActionType.BUY,
                            symbol=symbol,
                            quantity=qty,
                            reason="SMA20 breakout",
                        )
                    )

            elif prev_close >= prev_sma and price < curr_sma and ctx.has_position(symbol):
                actions.append(
                    Action(
                        type=ActionType.SELL,
                        symbol=symbol,
                        quantity=0,
                        reason="SMA20 breakdown",
                    )
                )

        return actions
```

### 5) Lifecycle hooks nên dùng

- `on_start(ctx)`: init state/log đầu phiên.
- `on_step(ctx)`: logic chính mỗi bar.
- `on_end(ctx)`: tổng kết trạng thái cuối phiên.

### 6) Checklist trước khi dùng thật

- Dữ liệu đủ `lookback` cho chỉ báo.
- Không vượt vốn (`qty > 0`, kiểm soát `allocation`).
- Có lý do lệnh rõ ràng trong `reason` để debug.
- Chạy backtest trên nhiều mã / nhiều giai đoạn.
- So sánh với benchmark (`BuyAndHoldBot`).

## Tương tác với profile

- Profile mẫu đặt tại `user/builtin/profiles/*.json`.
- Khi bot dùng profile, ưu tiên profile có cùng tên technique.
- Nếu không có profile, bot vẫn phải chạy được với confidence mặc định.

## Gợi ý quy trình phát triển

1. Prototype nhanh bằng notebook:
    - `notebooks/guide/backtest_tutorial.ipynb`
    - `notebooks/guide/profiler_tutorial.ipynb`
2. Đóng gói thành bot trong `user/bots/`.
3. Kiểm tra trên nhiều symbol/timeframe trước khi dùng cho quyết định thực tế.

## Phạm vi hiện tại

Thư mục này đã được liên kết runtime: code `*.py` trong `user/bots/` sẽ được quét và import tự động (bỏ qua file bắt đầu bằng `_`).
