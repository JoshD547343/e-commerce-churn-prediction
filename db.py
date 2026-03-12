import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "churn_predictions.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at                TEXT    NOT NULL,

            -- Demographics
            age                       REAL    NOT NULL,
            gender                    TEXT    NOT NULL,
            signup_quarter            TEXT    NOT NULL,
            membership_years          REAL    NOT NULL,

            -- Purchase behaviour
            total_purchases           REAL    NOT NULL,
            average_order_value       REAL    NOT NULL,
            days_since_last_purchase  REAL    NOT NULL,
            discount_usage_rate       REAL    NOT NULL,
            returns_rate              REAL    NOT NULL,
            cart_abandonment_rate     REAL    NOT NULL,
            wishlist_items            REAL    NOT NULL,

            -- Engagement
            login_frequency           REAL    NOT NULL,
            session_duration_avg      REAL    NOT NULL,
            pages_per_session         REAL    NOT NULL,
            email_open_rate           REAL    NOT NULL,
            mobile_app_usage          REAL    NOT NULL,
            social_media_engagement   REAL    NOT NULL,
            product_reviews_written   REAL    NOT NULL,

            -- Financial / support
            lifetime_value            REAL    NOT NULL,
            credit_balance            REAL    NOT NULL,
            customer_service_calls    REAL    NOT NULL,
            payment_method_diversity  REAL    NOT NULL,

            -- Engineered features
            engagement_recency_ratio  REAL    NOT NULL,
            purchase_value_score      REAL    NOT NULL,
            activity_score            REAL    NOT NULL,

            -- Result
            churn_prediction          INTEGER NOT NULL,
            churn_probability         REAL    NOT NULL,
            risk_level                TEXT    NOT NULL
        )
        """)
        conn.commit()


def insert_prediction(created_at, customer, engineered, prediction, probability, risk_level):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                created_at, age, gender, signup_quarter, membership_years,
                total_purchases, average_order_value, days_since_last_purchase,
                discount_usage_rate, returns_rate, cart_abandonment_rate, wishlist_items,
                login_frequency, session_duration_avg, pages_per_session, email_open_rate,
                mobile_app_usage, social_media_engagement, product_reviews_written,
                lifetime_value, credit_balance, customer_service_calls, payment_method_diversity,
                engagement_recency_ratio, purchase_value_score, activity_score,
                churn_prediction, churn_probability, risk_level
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            created_at,
            customer['Age'],                        customer['Gender'],
            customer['Signup_Quarter'],              customer['Membership_Years'],
            customer['Total_Purchases'],             customer['Average_Order_Value'],
            customer['Days_Since_Last_Purchase'],    customer['Discount_Usage_Rate'],
            customer['Returns_Rate'],                customer['Cart_Abandonment_Rate'],
            customer['Wishlist_Items'],              customer['Login_Frequency'],
            customer['Session_Duration_Avg'],        customer['Pages_Per_Session'],
            customer['Email_Open_Rate'],             customer['Mobile_App_Usage'],
            customer['Social_Media_Engagement_Score'], customer['Product_Reviews_Written'],
            customer['Lifetime_Value'],              customer['Credit_Balance'],
            customer['Customer_Service_Calls'],      customer['Payment_Method_Diversity'],
            engineered['engagement_recency_ratio'],
            engineered['purchase_value_score'],
            engineered['activity_score'],
            prediction, probability, risk_level
        ))
        conn.commit()


def fetch_latest(limit: int = 20):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """SELECT id, created_at, age, gender, churn_prediction,
               churn_probability, risk_level
               FROM predictions
               ORDER BY id DESC LIMIT ?""",
            (limit,)
        )
        return cur.fetchall()


def fetch_all():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT * FROM predictions ORDER BY id DESC")
        return cur.fetchall()


def delete_prediction(prediction_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        conn.commit()


def clear_all():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()