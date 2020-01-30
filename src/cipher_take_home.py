from typing import List, Tuple
from enigma.machine import EnigmaMachine
from faker import Faker
import re

from nlp_takehome.src.engine import Engine


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
    # solution here
    return cipher_list


def str_score(str_a: str, str_b: str) -> float:
    if len(str_a) != len(str_b):
        return 0

    n_correct = 0

    for a, b in zip(str_a, str_b):
        n_correct += int(a == b)

    return n_correct / len(str_a)


def score(predicted_plain: List[str], correct_plain: List[str]) -> float:
    correct = 0

    for p, c in zip(predicted_plain, correct_plain):
        if str_score(p, c) > 0.8:
            correct += 1

    return correct / len(correct_plain)


if __name__ == "__main__":
    engine = Engine(220000)
    engine.early_stopping()
    plain, cipher = generate_data(1 << 5)

    for i in range(len(plain)):
        print('>', cipher[i])
        print('=', plain[i])
        output_words, attentions = engine.evaluate(cipher[i])
        output_sentence = ''.join(output_words)
        print(f'< {output_sentence} \n')

    plain, cipher = generate_data(1 << 14)
    print(score(predict(cipher), plain))
