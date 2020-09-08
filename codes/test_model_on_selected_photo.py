import unittest

from model_production import load_trained_model, test_model_on_selected_photo

class FighterTestName(unittest.TestCase):
    """Test model on selected url.
    """

    def test_khabib_nurmagomedov(self):
        """Tests url of Khabib Nurmagomedov."""
        model, out_encoder = load_trained_model()
        url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSBbmAcIU9L8RqR0c4UhIzvPNxb04vNVj0MQw&usqp=CAU'
        fighter_name, probability = test_model_on_selected_photo(url, model, out_encoder)
        self.assertEqual(fighter_name, 'Khabib-Nurmagomedov')
        self.assertEqual(probability > 0.7, True)

    def test_conor_mcgregor(self):
        """Tests url of Conor McGregor."""
        model, out_encoder = load_trained_model()
        url = 'https://imagez.tmz.com/image/33/1by1/2016/11/15/33b9ebee31bc5646938781305ff0cfe7_xl.jpg'
        fighter_name, probability = test_model_on_selected_photo(url, model, out_encoder)
        self.assertEqual(fighter_name, 'Conor-McGregor')
        self.assertEqual(probability > 0.7, True)

unittest.main()