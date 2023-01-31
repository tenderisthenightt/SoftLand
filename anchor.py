import random

anchor = {'candy':["D:/k-digital/source/web_mk2/similarity/static/img/similarity/anchor/candy.jpg", '/static/1/img/anchor/candy.jpg', 0.35],\
    'table':["D:/k-digital/source/web_mk2/similarity/static/img/similarity/anchor/table.jpg", "/static/1/img/anchor/table.jpg", 0.42],\
        'chair':["D:/k-digital/source/web_mk2/similarity/static/img/similarity/anchor/chair.jpg", "/static/1/img/anchor/chair.jpg", 0.32],\
            'stick':["D:/k-digital/source/web_mk2/similarity/static/img/similarity/anchor/stick.jpg", "/static/1/img/anchor/stick.jpg", 0.4],\
                'fan':["D:/k-digital/source/web_mk2/similarity/static/img/similarity/anchor/fan.png", "/static/1/img/anchor/fan.png", 0.33]}
quiz = ['candy', 'table', 'chair', 'stick', 'fan']

def random_sim():
    q = random.choice(quiz)
    p_path = anchor[q][0]
    h_path = anchor[q][1]
    sim = anchor[q][2]
    return q, p_path, h_path, sim