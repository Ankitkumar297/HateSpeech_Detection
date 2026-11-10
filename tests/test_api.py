import csv
import io
import os
import tempfile
import unittest
from unittest.mock import patch


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(cls.temp_dir.name, "predictions.db")
        os.environ["ADMIN_API_KEY"] = "test-admin-key"
        from app import app
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_records_require_admin_key(self):
        for path in ("/api/history", "/api/admin/predictions", "/api/export/csv"):
            self.assertEqual(self.client.get(path).status_code, 403)
        self.assertEqual(self.client.delete("/api/admin/predictions/1").status_code, 403)
        response = self.client.get("/api/admin/predictions", headers={"X-Admin-Key": "test-admin-key"})
        self.assertEqual(response.status_code, 200)

    def test_csv_escapes_formula_cells(self):
        from database.db import insert_prediction
        insert_prediction("=1+1", "11", "Neutral Language", 50.0, {})
        response = self.client.get("/api/export/csv", headers={"X-Admin-Key": "test-admin-key"})
        rows = list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))
        self.assertTrue(any(row["User Text"] == "'=1+1" for row in rows))

    def test_unanalyzable_input_has_no_fake_confidence(self):
        response = self.client.post("/api/predict", json={"text": "!!!"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("confidence", response.get_json())

    def test_non_string_input_is_rejected(self):
        self.assertEqual(self.client.post("/api/predict", json={"text": 10}).status_code, 400)

    def test_everyday_identity_mentions_remain_neutral(self):
        from app import classifier
        for text in (
            "I met a man", "I met a men", "I met a woman", "I met a women",
            "I spoke with a woman yesterday", "The woman won the science award",
            "I do not hate women", "I met my neighbor for coffee",
        ):
            with self.subTest(text=text):
                self.assertEqual(classifier.predict(text)["prediction"], "Neutral Language")
        self.assertEqual(classifier.predict("I hate all women")["prediction"], "Hate Speech")
        self.assertEqual(classifier.predict("Women should not have rights")["prediction"], "Hate Speech")
        self.assertEqual(classifier.predict("You are an idiot")["prediction"], "Offensive Language")

    def test_pdf_ignores_client_claims(self):
        from app import classifier
        result = classifier.predict("A pleasant day at the park")
        with patch.object(classifier, "predict", return_value=result) as predict:
            response = self.client.post("/api/export/pdf", json={
                "text": "A pleasant day at the park",
                "prediction": "Hate Speech",
                "confidence": 100,
                "probabilities": [1, 2, 3],
                "id": "fake"
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "application/pdf")
            predict.assert_called_once_with("A pleasant day at the park")
        self.assertEqual(self.client.post("/api/export/pdf", json={"text": []}).status_code, 400)

    def test_history_page_bounds(self):
        headers = {"X-Admin-Key": "test-admin-key"}
        for value in ("-1", "0", "101", "bad"):
            self.assertEqual(self.client.get(f"/api/history?limit={value}", headers=headers).status_code, 400)
        self.assertEqual(self.client.get("/api/admin/predictions?offset=-1", headers=headers).status_code, 400)

    def test_deleted_records_are_not_reseeded(self):
        from database.db import get_db_connection, init_db
        conn = get_db_connection()
        conn.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        init_db()
        conn = get_db_connection()
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
