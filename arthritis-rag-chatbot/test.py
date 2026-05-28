from backend.rag_engine import (
    ask_question
)

while True:

    question = input(
        "\nYou: "
    )

    if question == "exit":
        break

    response = (
        ask_question(
            question
        )
    )

    print(
        "\nAssistant:",
        response
    )