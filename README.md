# RFISUBCODDES-Predictor

Of course! Let's break this down like we're explaining it to a friend who doesn't know anything about coding or data.

### What is this project, in one sentence?

**It's a smart computer program that tells an airline which extra-paid services (like seat upgrades or extra baggage) will make them the most money in any specific city and month.**

---

### Let's use a simple analogy: A Lemonade Stand

Imagine you run a lemonade stand. You don't just sell lemonade; you also sell **cookies**, **chips**, and **umbrellas** for shade (these are your "ancillary products").

*   **On a hot, sunny day** at the park, you sell a lot of **lemonade and umbrellas**.
*   **On a cool, cloudy day** outside a school, you sell more **cookies and chips**.

After a while, you learn the patterns. Your brain becomes a "model" that predicts what to sell and where.

This project does the **exact same thing**, but for a huge airline. It learned the patterns from years of sales data to predict what will sell best.

---

### What "Extra Services" are we talking about?

The program looks at 22 different things an airline sells. The main ones are:
*   **UPGRADE** (e.g., moving from Economy to Business Class)
*   **PRE PAID BAGGAGE** (paying for baggage online before the flight)
*   **EXCESS BAGGAGE** (paying for overweight bags at the airport)
*   **EXCESS PIECE** (paying for an extra bag)
*   **SEAT ASSIGNMENT** (paying to pick your seat early)

---

### How does the program work? (Step-by-Step)

**Step 1: Learning from the Past**
The program read a giant spreadsheet with **150,000+ rows** of sales history from 2015 to 2020. Each row was a transaction: "We sold 2 seat upgrades in Sri Lanka in January 2015 for $1,223."

**Step 2: Cleaning up the Data**
It threw out any messy or incomplete rows (like if the price was missing), so it only learned from good, clean data.

**Step 3: Finding the "Best Seller"**
For every single city and month combination, it calculated which service made the most money **per transaction**. This is the key!
*   Example: Selling one **Upgrade** for $600 is better than selling four **Baggage** fees for $100 each ($400 total), even though you sold more items.

**Step 4: The Computer "Studys" the Patterns**
A very smart algorithm called **XGBoost** looked at all this data and learned the hidden patterns all by itself. It figured out things like:
*   "Ah, in **Hong Kong** during **Summer** (June, July, August), people are willing to pay a lot for **Upgrades**."
*   "I see that in **India** in **December** (holiday season), people are bringing lots of luggage, so **Extra Baggage** is the most profitable."

**Step 5: It Becomes a Prediction Machine**
Now, the program is trained. You can ask it a question, and it will give you an answer based on what it learned.

---

### Let's see it in action!

You ask the program: **"What should we promote in Sri Lanka in June 2023?"**

The program doesn't guess. It calculates and gives you its top 3 recommendations, with a confidence score:

1.  **UPGRADE** (94% confidence)
2.  **PRE PAID BAGGAGE** (4% confidence)
3.  **PRE-RESERVED SEAT** (1% confidence)

This means the program is **94% sure** that promoting Upgrades will make the airline the most money per customer in Sri Lanka that month.

---

### What will the Website (Streamlit App) look like?

The Streamlit app is the **simple, visual website** that lets anyone use this powerful program.

**You would see and do this:**

1.  **See Pretty Charts:** As soon as you open the page, you'd see graphs showing sales trends over time, a list of the most popular services, and which cities spend the most money. This gives you a quick overview.

2.  **Use the Prediction Tool:** This is the main part. You'd see:
    *   A dropdown menu to **select a city** (like "Sri Lanka" or "Singapore").
    *   Sliders to **choose a year and month**.
    *   A big button that says **"PREDICT"**.

3.  **Get Your Results:** After you click the button, the answers appear right away in a clear, easy-to-read list and a simple bar chart, just like the example above.

4.  **Check the Accuracy:** There might be a section that says, "Our predictions are correct **69%** of the time." This builds trust so you know the advice is reliable.

### Why is this so useful for the airline?

*   **Maximize Profit:** They can stop guessing and start using data to make decisions, ensuring they make the most money possible from every flight.
*   **Targeted Marketing:** Instead of showing every passenger an ad for every service, they can show passengers in Sri Lanka an ad for Upgrades, and passengers in London an ad for Extra Baggage. This is more effective.
*   **Save Time and Effort:** Managers don't need to spend weeks analyzing spreadsheets. The program does the hard work instantly.

