import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Data for products sold in November 2024
data = '''product_name  total_quantity_sold
Bread                237.0
Cereal               173.0
Butter               132.0
Juice                 97.0
Eggs                 70.0
Coffee               49.0
Tea                  47.0
Milk                 44.0
Yogurt               21.0'''

# Read the data into a DataFrame
df = pd.read_csv(StringIO(data), sep="\s+")

# Create the histogram
plt.figure(figsize=(10, 6))
plt.bar(df['product_name'], df['total_quantity_sold'], color='skyblue')
plt.title('Counts of Products Sold in November 2024')
plt.xlabel('Product Name')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
