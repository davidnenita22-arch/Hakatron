"""Spain – ¡Feliz Navidad!  Rebuild the office Christmas board.

Usage: python3 solve.py   (reads purchases.csv + office_price_guide.csv, writes board.png)

Board = pilots as columns (1..21) x recommended-list number as rows (1..6, top to bottom).
A cell is filled when that pilot has at least one receipt for that item at a fair
price per unit (Fair_min_eur <= Total_paid_eur / Quantity <= Fair_max_eur).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

purchases = pd.read_csv("purchases.csv")
guide = pd.read_csv("office_price_guide.csv")

# 1) Office rule: price per unit inside the fair range (inclusive)
df = purchases.merge(guide, on="Product", how="left")
df["price_per_unit"] = df.Total_paid_eur / df.Quantity
df["fair"] = df.price_per_unit.between(df.Fair_min_eur, df.Fair_max_eur)
fair = df[df.fair]
print(f"receipts: {len(df)}  fair: {len(fair)}  fake (too cheap): "
      f"{(df.price_per_unit < df.Fair_min_eur).sum()}  tourist trap: "
      f"{(df.price_per_unit > df.Fair_max_eur).sum()}")

# 2) Board: rows = List_no 1..6, columns = Pilot_no 1..21
board = (pd.crosstab(fair.List_no, fair.Pilot_no)
         .reindex(index=range(1, 7), columns=range(1, 22), fill_value=0) > 0)

for no, row in board.iterrows():
    print(f"{no} {guide.set_index('List_no').Product[no]:<10} " + "".join("#" if v else "." for v in row))

# 3) Figure
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.imshow(board.values, cmap="Greys", aspect="equal")
ax.set_xticks(range(21), [str(c) for c in board.columns])
ax.set_yticks(range(6), [f"{n}. {guide.set_index('List_no').Product[n]}" for n in board.index])
ax.set_xlabel("Pilot_no")
ax.set_title("Christmas board: fair-price souvenirs per pilot  ->  cRuCiO = CRUCIO")
ax.set_xticks([x - 0.5 for x in range(22)], minor=True)
ax.set_yticks([y - 0.5 for y in range(7)], minor=True)
ax.grid(which="minor", color="0.85", lw=0.5)
ax.tick_params(which="minor", length=0)
plt.tight_layout()
plt.savefig("board.png", dpi=120)

word = "CRUCIO"
print("WORD:", word)
print("FLAG:", f"SIGMOID_{word}")
