from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import URL, DataRequired

class URLRequisition(FlaskForm):
    url = StringField('url', validators=[URL(message='Invalid URL'),
                                         DataRequired()])
    submit = SubmitField('Predict Fighters')


