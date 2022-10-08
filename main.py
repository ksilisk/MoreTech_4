from parser import Parser
import json


# buh_ru id = 4830
# klerk id = 7300

def main():
    parser = Parser()
    back_data = {}
    with open('result.json', 'r') as f:
        back_data = json.loads(f.read())
    f.close()
    new_data = {'buh.ru': parser.buh_ru(4830, 50), 'klerk.ru': parser.klerk_ru(7300, 0)}
    for key in back_data.keys():
        new_data[key].extend(back_data[key])
    with open('result.json', 'w') as f:
        json.dump(new_data, f, ensure_ascii=False)
    f.close()


if __name__ == '__main__':
    main()
