from twilio.rest import Client

client=Client('ACb1edc646cadb42b6dd179151b754a079','898cc1bf6c1d0680ca471854c359448e')
def dummy_fn():
    message= client.messages.create(
        body="Your OTP is 12345",
        from_='+19723629969',
        to='+919003666098'
    )
    print(message.body)