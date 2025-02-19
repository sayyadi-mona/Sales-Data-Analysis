import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import ttest_rel

print("\n\n\033[1m✅ Loading full dataset...\033[0m")
sales_data = pd.read_csv(r"D:\Desktop\Mona\Data\synthetic_beverage_sales_data.csv")
print("\n\n\033[1m✅ Data loaded successfully\033[0m")
print(sales_data.head())

sales_data["Order_Date"] = pd.to_datetime(sales_data["Order_Date"])
sales_data["Year"] = sales_data["Order_Date"].dt.year
sales_data["Month"] = sales_data["Order_Date"].dt.month

season_map = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}

sales_data["Season"] = sales_data["Month"].map(season_map)

print("\n\n\033[1m✅ Seasonal Sales Analysis:\033[0m")
seasonal_sales = sales_data.groupby("Season")["Quantity"].sum().reset_index()
print(seasonal_sales.head())

seasonal_sales.to_csv(r"D:\Desktop\Mona\Data\seasonal_sales_analysis.csv", index=False)

plt.figure(figsize=(8, 5))
plt.pie(seasonal_sales["Quantity"], labels=seasonal_sales["Season"], autopct='%1.1f%%', colors=sns.color_palette("coolwarm", len(seasonal_sales)))
plt.title("Sales Distribution by Season")
plt.savefig(r"D:\Desktop\Mona\Data\seasonal_sales_distribution.png")
plt.show()

sales_per_Product = sales_data.groupby(["Product", "Year"])["Quantity"].sum().unstack(level="Year").reset_index()

Year_of_Sales = sales_data["Year"].unique()
for year in Year_of_Sales:
    sales_of_Year = sales_data[sales_data["Year"] == year]["Quantity"].sum()
    sales_per_Product[f"Share in {year}"] = sales_per_Product[year].map(lambda q: round((q / sales_of_Year) * 100, 2))

sales_per_Product = sales_per_Product.rename(columns={2021: "2021", 2022: "2022", 2023: "2023"})
sales_per_Product = sales_per_Product.sort_values(by="2023", ascending=False)

print("\n\n\033[1m✅ Creating cumulative share...\033[0m")
for col in sales_per_Product.columns:
    if "Share" in col:
        sales_per_Product[f"Cumulative {col}"] = sales_per_Product[col].cumsum()

print(sales_per_Product.head())

sales_per_Product.to_csv(r"D:\Desktop\Mona\Data\yearly_product_sales.csv", index=False)

sales_in_time_series = sales_data.groupby(["Month", "Year"])["Quantity"].sum()

sales_in_time_series = sales_in_time_series.map(lambda q: round(q / 1000, 0))

sales_in_time_series = sales_in_time_series.unstack(level="Year")
sales_in_time_series = sales_in_time_series.reset_index()

print("\n\033[1m✅ Sales Time Series Preview:\033[0m")
print(sales_in_time_series.head(12))

sales_in_time_series.to_csv(r"D:\Desktop\Mona\Data\monthly_sales_trend.csv", index=False)

plt.figure(figsize=(10, 5))
for i in sales_in_time_series.columns[1:]:
    plt.plot(sales_in_time_series["Month"], sales_in_time_series[i], marker="o", linestyle="-")

plt.xlabel("Month")
plt.ylabel("Sales (in thousands)")
plt.title("Monthly Sales Trends Over Different Years")
plt.legend(sales_in_time_series.columns[1:], title="Year")
plt.grid(True)

plt.savefig(r"D:\Desktop\Mona\Data\monthly_sales_trend.png")

plt.show()

print("\n\033[1m✅ Sales Per Discount Data:\033[0m")

sales_per_discount = sales_data.groupby(["Product", "Discount"])["Quantity"].sum()

sales_per_discount = sales_per_discount.unstack(level="Discount").reset_index()

sales_per_discount = sales_per_discount.rename(columns={0.00: "Without Discount"})

sales_per_discount.to_csv(r"D:\Desktop\Mona\Data\sales_per_discount.csv", index=False)

print(sales_per_discount.head())

discount_levels = sales_per_discount.columns[2:]
t_test_result = {}

for discount in discount_levels:
    t_stat, p_value = ttest_rel(
        sales_per_discount["Without Discount"],
        sales_per_discount[discount]
    )
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    t_test_result[discount] = {"T_stat": t_stat, "P_value": p_value, "Significance": significance}

t_test_result_df = pd.DataFrame(t_test_result).T

t_test_result_df.to_csv(r"D:\Desktop\Mona\Data\discount_ttest_results.csv", index=False)

print("\n\033[1m✅ T-test Results for Discount Impact:\033[0m")
print(t_test_result_df)


channels_share = sales_data.groupby(["Customer_Type", "Category"])["Quantity"].sum()

channels_share = channels_share.unstack(level="Customer_Type").reset_index()

sales_channels = sales_data["Customer_Type"].unique()

for channel in sales_channels:
    total_sales_per_category = sales_data[sales_data["Customer_Type"] == channel]["Quantity"].sum()
    channels_share[channel] = channels_share[channel].map(lambda q: round((q / total_sales_per_category) * 100, 2))

channels_share.to_csv(r"D:\Desktop\Mona\Data\customer_type_category_share.csv", index=False)

print("\n\033[1m✅ Customer Type vs. Category Sales Share (Top 10 Rows):\033[0m")
print(channels_share.head(10))

plot_columns = channels_share.columns[1:]
fig, axes = plt.subplots(1, len(plot_columns), figsize=(12, 4))

for ax, column in zip(axes, plot_columns):
    ax.pie(channels_share[column], labels=channels_share["Category"], autopct='%1.0f%%', startangle=90)
    ax.set_title(f"Share of Channels - {column}")

plt.savefig(r"D:\Desktop\Mona\Data\customer_type_category_share.png")
plt.show()

print("\n\033[1m✅ Customer Loyalty Analysis...\033[0m")
Loyal_Customers = sales_data[sales_data["Year"] == 2023].groupby(["Customer_ID", "Region", "Order_ID"])["Total_Price"].agg(["count", "sum"]).reset_index()
Loyal_Customers = Loyal_Customers.rename(columns={"count": "Row of Invoice", "sum": "Sum Value of Order"})
Loyal_Customers.to_csv(r"D:\Desktop\Mona\Data\loyal_customers.csv", index=False)
print(Loyal_Customers.head())

print("\n\033[1m✅ Customer Behavior Analysis...\033[0m")
Customer_Invoice_Analyse = Loyal_Customers.groupby(["Customer_ID", "Region"])["Sum Value of Order"].agg(["count", "sum"]).reset_index()
Customer_Invoice_Analyse = Customer_Invoice_Analyse.rename(columns={"count": "Sales Repetition", "sum": "Sum Value of Order"})
Customer_Invoice_Analyse.to_csv(r"D:\Desktop\Mona\Data\customer_invoice_analysis.csv", index=False)
print(Customer_Invoice_Analyse.head())

purchase_distribution = Customer_Invoice_Analyse["Sales Repetition"].value_counts().reset_index()
purchase_distribution.columns = ["Sales Repetition", "Number of Customers"]

plt.figure(figsize=(12, 6))
sns.scatterplot(data=purchase_distribution, x="Sales Repetition", y="Number of Customers", alpha=0.7, edgecolor="black")
plt.title("Customer Purchase Frequency Distribution", fontsize=14)
plt.xlabel("Number of Purchases in 2023", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.grid(True)
plt.savefig(r"D:\Desktop\Mona\Data\purchase_frequency_distribution.png")
plt.show()

print("\n\033[1m✅ Quartile Calculation...\033[0m")
q1 = Customer_Invoice_Analyse["Sales Repetition"].quantile(0.25)
q3 = Customer_Invoice_Analyse["Sales Repetition"].quantile(0.75)
print(f"First Quartile (Q1 - 25%): {q1}")
print(f"Third Quartile (Q3 - 75%): {q3}")

print("\n\033[1m✅ Categorizing Customer Behavior...\033[0m")
def Customer_behavior(frequency):
    if frequency < q1:
        return "Low Activity"
    elif q1 <= frequency <= q3:
        return "Loyal Customer"
    else:
        return "High Activity"

Customer_Invoice_Analyse["Customer Behavior"] = Customer_Invoice_Analyse["Sales Repetition"].apply(Customer_behavior)
Customer_Invoice_Analyse.to_csv(r"D:\Desktop\Mona\Data\customer_behavior_classification.csv", index=False)
print(Customer_Invoice_Analyse.head())

Loyal_Customers = sales_data[sales_data["Year"] == 2023].groupby(["Customer_ID", "Order_ID", "Discount"])["Total_Price"].agg(["count", "sum"])
Loyal_Customers = Loyal_Customers.rename(columns={"count": "Row of Invoice", "sum": "Sum Value of Order"})
Loyal_Customers = Loyal_Customers.reset_index()
Loyal_Customers = Loyal_Customers.sort_values(by=["Customer_ID", "Sum Value of Order"], ascending=False)

Customer_Invoice_Analyse = Loyal_Customers.groupby(["Customer_ID", "Discount"])["Sum Value of Order"].agg(["count", "sum"])
Customer_Invoice_Analyse = Customer_Invoice_Analyse.rename(columns={"count": "Sales Repetition", "sum": "Sum Value of Order"})
Customer_Invoice_Analyse = Customer_Invoice_Analyse.reset_index()
Customer_Invoice_Analyse = Customer_Invoice_Analyse.sort_values(by="Sales Repetition", ascending=True)

Customer_Invoice_Analyse["Customer Behavior"] = Customer_Invoice_Analyse["Sales Repetition"].apply(Customer_behavior)
Customer_Behavior_per_Region = Customer_Invoice_Analyse.groupby(["Discount", "Customer Behavior"])["Customer_ID"].count()
Customer_Behavior_per_Region = Customer_Behavior_per_Region.unstack(level="Discount").reset_index()

Customer_Behavior_per_Region.to_csv(r"D:\Desktop\Mona\Data\customer_behavior_per_region.csv", index=False)

print("\n\033[1m✅ Customer Behavior Analysis by Discount Level:\033[0m")
print(Customer_Behavior_per_Region)

print("\n\033[1m🎉 Sales Analysis Enhanced Completed!\033[0m")