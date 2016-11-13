
#ifndef MYTEST_H
#define MYTEST_H

class MyTest : public Test
{
public:
	MyTest()
	{
		m_world->SetGravity(b2Vec2(0, 0));

		b2PolygonShape shape;
		shape.SetAsBox(10, 1);

		b2BodyDef bdef;
		b2Body *body = m_world->CreateBody(&bdef);
		body->CreateFixture(&shape, 1);

		b2CircleShape circle;
		circle.m_radius = 3;

		///////////

		bdef.position.Set(0, 20);
		bdef.type = b2_dynamicBody;
		b2Body *body2 = m_world->CreateBody(&bdef);
		shape.SetAsBox(2, 2);
		body2->CreateFixture(&circle, 1);

		bdef.position.Set(0, 20);
		bdef.type = b2_dynamicBody;
		b2Body *body3 = m_world->CreateBody(&bdef);
		shape.SetAsBox(2, 2,b2Vec2(7, 5), 0);
		body3->CreateFixture(&shape, 1);

		b2MotorJointDef wj;
		wj.bodyA = body2;
		wj.bodyB = body3;


		m_world->CreateJoint(&wj);
	}

	static Test* Create()
	{
		return new MyTest;
	}
};

#endif
