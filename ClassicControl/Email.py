import smtplib, ssl


class email():
    def __init__(self, message=None):
        self.port = 465  # For SSL
        self.password = 'bwct igzx uyvl gqsb'
        self.message = message if not None else "Subject: Yo. Your NN is done running."

    def send(self):
        smtp_server = "smtp.gmail.com"
        sender_email = "maxtang2001@gmail.com"
        receiver_email = "maxtang2001@gmail.com"
        password = 'bwct igzx uyvl gqsb'
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL(smtp_server, self.port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, self.message)