import re
from typing import List, Tuple

from enigma.machine import EnigmaMachine
from faker import Faker

from evaluation_engine import EvaluationEngine
from levenshtein_distance import levenshtein_distance


class ConfiguredMachine:
    def __init__(self):
        self.machine = EnigmaMachine.from_key_sheet(
            rotors='II IV V',
            reflector='B',
            ring_settings=[1, 20, 11],
            plugboard_settings='AV BS CG DL FU HZ IN KM OW RX')

    def reset(self):
        self.machine.set_display('WXC')

    def encode(self, plain_str: str) -> str:
        self.reset()
        return self.machine.process_text(plain_str)

    def batch_encode(self, plain_list: List[str]) -> List[str]:
        encoded = list()
        for s in plain_list:
            encoded.append(self.encode(s))
        return encoded


def pre_process(input_str):
    return re.sub('[^a-zA-Z]', '', input_str).upper()


def generate_data(batch_size: int, seq_len: int = 42) -> Tuple[List[str], List[str]]:
    fake = Faker()
    machine = ConfiguredMachine()

    plain_list = fake.texts(nb_texts=batch_size, max_nb_chars=seq_len)
    plain_list = [pre_process(p) for p in plain_list]
    cipher_list = machine.batch_encode(plain_list)
    return plain_list, cipher_list


def predict(cipher_list: List[str]) -> List[str]:
    evaluation_engine = EvaluationEngine()
    return [evaluation_engine.evaluate(cipher) for cipher in cipher_list]


def average_levenshtein(predicted_plain: List[str], correct_plain: List[str]) -> float:
    correct = 0

    for predicted, plain in zip(predicted_plain, correct_plain):
        correct += levenshtein_distance(plain, predicted)
    return correct / len(correct_plain)


if __name__ == "__main__":
    plain, cipher = generate_data(1000)
    print(average_levenshtein(predict(cipher), plain))
